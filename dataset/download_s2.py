import contextlib
import inspect
import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

import keyring
import pwinput
import requests
import urllib3.exceptions
from tqdm.autonotebook import tqdm

# Define the path to the configuration file
config_file_path = "config.json"

# Default configuration values
default_config = {
    "destination": "output",
    "start_date_str": "2020-01-01",
    "end_date_str": "2020-01-31",
    "spatial_filter": "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((-140.99778 41.6751050889,-140.99778 83.23324,-52.6480987209 41.6751050889,-52.6480987209 83.23324,-140.99778 41.6751050889))')",
    "product_filter": "contains(Name,'S3A_SL_2_LST')",
    "serviceName": "odata_dataspace",
    "download_attempts": 10
}

# Load the configuration
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
else:
    # If the config file doesn't exist, create it with default values
    with open(config_file_path, 'w') as config_file:
        json.dump(default_config, config_file, indent=4)
    config = default_config

# %% configuration
destination = Path(config["destination"])
start_date_str = config["start_date_str"]
end_date_str = config["end_date_str"]
spatial_filter = config["spatial_filter"]
product_filter = config["product_filter"]
serviceName = config["serviceName"]
download_attempts = config["download_attempts"]

# %% setup
if not os.path.exists(destination):
    Path(destination).mkdir(parents=True, exist_ok=True)

# %% functions

def authenticate(username=None):
    refresh_token = None
    authenticated = False
    while not authenticated:
        if username is None:
            username = input("Please enter username:")

        if keyring.get_password(serviceName, username) is None:
            keyring.set_password(serviceName, username, pwinput.pwinput("Please enter your o-data password:"))
        try:
            refresh_token = get_refresh_token(username, keyring.get_password(serviceName, username))
        except ConnectionRefusedError:
            keyring.delete_password(serviceName, username)
            continue
        else:
            print(f"Authenticated as {username}")
            authenticated = True

    return username, refresh_token


def get_refresh_token(username, password):
    # Define the endpoint and parameters
    url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password,
        'client_id': 'cdse-public'
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code != 200:
        raise ConnectionRefusedError("Error getting token: {}".format(response.json()))
    else:
        return response.json()['refresh_token']


def get_access_token(user, refresh_token, refresh_count=0):
    # Define the endpoint and parameters
    url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': 'cdse-public'
    }

    post_response = requests.post(url, headers=headers, data=data)

    if post_response.status_code != 200:
        if post_response.json()['error'] == 'invalid_grant':
            print("Invalid grant, trying to get new refresh token")
            if refresh_count > 10:
                raise ConnectionRefusedError("Error getting token: {}".format(post_response.json()))

            refresh_token = authenticate(user)[1]

            return get_access_token(user, refresh_token, refresh_count + 1)

        else:
            raise ConnectionRefusedError("Error getting token: {}".format(post_response.json()))

    else:
        return post_response.json()['access_token'], refresh_token


def rename_file(file_path, new_extension):
    try:
        new_file_name = get_new_file_name(file_path, new_extension)
        os.rename(file_path, new_file_name)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except FileExistsError:
        print(f"Error: File with new name {new_file_name} already exists.")
        return None
    else:
        return new_file_name


def get_new_file_name(file_path, new_extension):
    # Split the file path into the base name and the extension
    base_name, current_extension = os.path.splitext(file_path)

    new_file_name = base_name + new_extension
    return new_file_name

# %% Authentication
auth_user = None
if os.path.exists("username.txt"):
    # Open the file in read mode
    with open("username.txt", 'r') as file:
        # Read the first line and store it in the 'username' variable
        auth_user = file.readline().strip()

auth_user, auth_refresh_token = authenticate(auth_user)

with open("username.txt", 'w') as file:
    file.write(auth_user)

# %% Query the API

# Convert strings to datetime objects
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

# Calculate the end date by adding one day
end_date += timedelta(days=1)

# Format the date range in the specified format
temporal_filter = (
    f"ContentDate/Start gt {start_date.isoformat()}Z "
    f"and ContentDate/Start lt {end_date.isoformat()}Z"
)

api_query = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
    f"$filter={spatial_filter} and {product_filter} and {temporal_filter}&$top=50&$orderby=ContentDate/Start desc"
)

products = []

pbar = tqdm(desc="Scanning Files", unit=" files")
depth = 0

lastValue = 0
while True:
    pbar.update(len(products) - lastValue)
    lastValue = len(products)

    depth += 1

    data = requests.get(api_query).json()

    if "value" in data:
        products.extend(data['value'])

    if "@odata.nextLink" in data:
        api_query = data['@odata.nextLink']
    else:
        print("Found no more pages.")
        break

pbar.close()
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")


# %% Download the products

def download_product(product_id, access_token, filename, download_chunk_size=8192):
    try:
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {access_token}"}
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, headers=headers, stream=True)
        content_size = int(response.headers.get('Content-Length', 0))

        progress_bar = tqdm(desc=f"Downloading {filename}", total=content_size, unit='B', unit_scale=True,
                            unit_divisor=1024, leave=False, miniters=1)
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=download_chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
    except urllib3.exceptions.ProtocolError as e:
        # catch when the connection drops
        return False, e
    except requests.exceptions.ChunkedEncodingError as e:
        # catch when server stops sending chunks... Token expired?
        return False, e
    else:
        return True, None


for product in tqdm(products, desc="Downloading Products", unit="product"):
    auth_access_token, auth_refresh_token = get_access_token(auth_user, auth_refresh_token)
    productID = product['Id']
    productName = product['Name']

    finalProductName = get_new_file_name(productName, ".zip")

    if Path(Path(destination) / Path(finalProductName)).resolve().exists():
        print("File already downloaded... skipping download.")
        continue

    if Path.exists(productName):
        os.remove(productName)

    downloaded = False
    downloadAttempt = 0
    while not downloaded:

        if downloadAttempt > download_attempts:
            print(f"Download attempt failed more than {download_attempts} times for product with ID {productID}. "
                  f"Moving to next product")
            break

        downloaded, ex = download_product(productID, auth_access_token, productName)

        if not downloaded:
            print(f"Download failed for product {productID}{str(ex)}. Trying again...")

        downloadAttempt += 1

    fileName = rename_file(productName, ".zip")
    if fileName is not None:
        shutil.move(fileName, Path(destination).resolve())

# %%