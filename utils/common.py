import yaml
import time
import wandb
import pandas as pd
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


# get configs for a sweep from .yaml file
def get_configs_from_file(path_yaml):
    dict_yaml = yaml.load(open(path_yaml).read(), Loader=yaml.Loader)
    sweep_config = dict_yaml["sweep_config"]
    params_config = dict_yaml["params_config"]
    search_space = {}
    hash_keys = []
    for k, v in params_config.items():
        search_space[k] = {"values": v}
        if len(v) > 1:
            hash_keys.append(k)
        if k == "num_runs":
            assert int(v[0]) > 0
            search_space["runs"] = {"values": list(range(int(v[0])))}
    search_space["hash_keys"] = {"values": [hash_keys]}
    sweep_config["parameters"] = search_space
    return sweep_config


# modify some specific hyper parameters in sweep's config
def modify_sweep(sweep_config, dict_new):
    for key in dict_new.keys():
        sweep_config["parameters"][key] = {"values": dict_new[key]}
    return sweep_config


def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        Run_time = end_time - begin_time
        print(
            "Execution time for func [%s] is [%s]" % (str(func.__name__), str(Run_time))
        )
        return ret

    return call_func


def get_timestamp():
    time.tzset()
    now = int(round(time.time() * 1000))
    timestamp = time.strftime("%Y-%m%d-%H%M", time.localtime(now / 1000))
    return timestamp


# calculate the size of a sweep's search space or the number of runs
def count_sweep(mode, entity, project, id):
    # mode: size_space, num_runs
    api = wandb.Api()
    sweep = api.sweep("%s/%s/%s" % (entity, project, id))
    if mode == "size_space":
        cnt = 1
        params = sweep.config["parameters"]
        for key in params.keys():
            cnt *= len(params[key]["values"])
    elif mode == "num_runs":
        cnt = len(sweep.runs)
    return cnt


def create_comp_csv(y_true, y_pred, classes, filepath):
    """
    Create a CSV file containing true and predicted values for each class.

    Args:
        y_true (numpy.ndarray): True labels, shape (N, C).
        y_pred (numpy.ndarray): Predicted labels, shape (N, C).
        classes (list): List of class names.
        filepath (str): Path to save the CSV file.
    """
    num_samples = y_true.shape[0]
    data = {"SampleID": np.arange(num_samples)}

    # Add true and predicted values for each class
    for i, class_name in enumerate(classes):
        data[f"True_{class_name}"] = y_true[:, i]
        data[f"Pred_{class_name}"] = y_pred[:, i]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def evaluate_model(sp_output_csv, classes):
    """
    Evaluate the model's performance using the output CSV files.

    Args:
        sp_output_csv (str): Path to the superpixel output CSV file.
        classes (list): List of class names.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, f1_score

    # Load superpixel outputs
    sp_df = pd.read_csv(sp_output_csv)

    # Extract species proportion predictions and true values
    sp_true = sp_df[
        [f"True_{cls}" for cls in classes]
    ].values  # Shape: (num_samples, num_classes)
    sp_pred = sp_df[
        [f"Pred_{cls}" for cls in classes]
    ].values  # Shape: (num_samples, num_classes)

    # Compute leading species indices
    true_leading = sp_true.argmax(axis=1).astype(
        int
    )  # 1D array of shape (num_samples,)
    pred_leading = sp_pred.argmax(axis=1).astype(
        int
    )  # 1D array of shape (num_samples,)

    # List of class indices
    class_indices = np.arange(len(classes))

    # Compute Overall Accuracy of Leading Species
    oa_leading_species = accuracy_score(true_leading, pred_leading)

    # Compute F1 Score of Leading Species
    f1_leading_species = f1_score(
        true_leading, pred_leading, labels=class_indices, average="macro"
    )
    wf1_leading_species = f1_score(
        true_leading, pred_leading, labels=class_indices, average="weighted"
    )

    # Compute Confusion Matrix
    cm = confusion_matrix(true_leading, pred_leading, labels=class_indices)

    # Compute Overall R² Score
    sp_pred_rounded = sp_pred.round(2)
    all_r2 = r2_score(sp_true.flatten(), sp_pred_rounded.flatten())

    # Compute R² Score per Species
    species_r2_scores = {}
    for i, species in enumerate(classes):
        r2 = r2_score(sp_true[:, i], sp_pred_rounded[:, i])
        species_r2_scores[species] = r2

    # Compile evaluation results
    evaluation_results = {
        "Overall Accuracy of Leading Species": oa_leading_species,
        "F1 Score of Leading Species": f1_leading_species,
        "Weighted F1 Score of Leading Species": wf1_leading_species,
        "Confusion Matrix": cm,
        "Overall R2 Score": all_r2,
        "R2 Scores per Species": species_r2_scores,
    }

    return evaluation_results, sp_df


def generate_eva(outputs, classes, output_dir):
    # Access the stored tensors
    preds_all = outputs["preds_all"]
    true_labels_all = outputs["true_labels_all"]

    # Convert tensors to NumPy arrays
    preds_all_np = preds_all.detach().cpu().numpy()
    true_labels_all_np = true_labels_all.detach().cpu().numpy()

    # Create CSV files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save superpixel outputs
    sp_output_csv = os.path.join(output_dir, "best_sp_outputs.csv")
    create_comp_csv(
        y_true=true_labels_all_np,
        y_pred=preds_all_np,
        classes=classes,
        filepath=sp_output_csv,
    )

    # Compute metrics using Evaluation class or function
    evaluation_results, sp_df = evaluate_model(
        sp_output_csv=sp_output_csv,
        classes=classes,
    )

    # Print or log evaluation results
    print("Evaluation Results:")
    print(
        f"Overall Accuracy of Leading Species: {evaluation_results['Overall Accuracy of Leading Species']:.4f}"
    )
    print(
        f"F1 Score of Leading Species: {evaluation_results['F1 Score of Leading Species']:.4f}"
    )
    print(
        f"Weighted F1 Score of Leading Species: {evaluation_results['Weighted F1 Score of Leading Species']:.4f}"
    )
    print("Confusion Matrix of Leading Species:")
    cm_df = pd.DataFrame(
        evaluation_results["Confusion Matrix"], index=classes, columns=classes
    )
    print(cm_df)
    print(f"Overall R2 Score: {evaluation_results['Overall R2 Score']:.4f}")
    print("R2 Scores per Species:")
    for species, r2 in evaluation_results["R2 Scores per Species"].items():
        print(f"{species}: {r2:.4f}")
    return sp_df


class PointCloudLogger(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Log only for the first batch in validation
        if batch_idx == 0:
            wandb_logger = trainer.logger.experiment  # Access WandbLogger
            n = min(4, len(batch["point_cloud"]))  # Handle smaller batches
            point_clouds = [pc.cpu().numpy() for pc in batch["point_cloud"][:n]]
            labels = [label.cpu().numpy() for label in batch["label"][:n]]
            # Captions for point clouds
            captions_1 = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(
                    labels[:n], outputs[0][:n].detach().cpu().numpy()
                )
            ]
            # Log point clouds
            wandb_logger.log(
                {"point_cloud": [wandb.Object3D(pc) for pc in point_clouds]},
                caption=captions_1,
            )

            # Log images
            images = [
                img.cpu().numpy().transpose(1, 2, 0) for img in batch["images"][:n]
            ]
            per_pixel_labels = [
                lbl.cpu().numpy() for lbl in batch["per_pixel_labels"][:n]
            ]
            captions = [
                f"Image Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(
                    per_pixel_labels[:n], outputs[1][:n].detach().cpu().numpy()
                )
            ]
            wandb_logger.log_image(key="sample_images", images=images, caption=captions)
