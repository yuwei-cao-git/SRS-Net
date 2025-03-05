import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .ResUnet import ResUnet

# -----------------------------------------------------------------------------------
# Parts of the fusion module
# -----------------------------------------------------------------------------------

class ConvBatchNormAct_x2(nn.Module):
    def __init__(self, n_filters, activation=F.leaky_relu, spatial_dropout=0, inference_dropout=False):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.dropout = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self, n_inputs, n_filters, activation=nn.LeakyReLU(), spatial_dropout=0, inference_dropout=False):
        super().__init__()
        self.n_filters = n_filters
        self.activation = activation
        self.spatial_dropout = spatial_dropout
        self.inference_dropout = inference_dropout

        # Convolutional branches for each input (assuming same channels)
        self.conv_branches = nn.ModuleList([
            ConvBatchNormAct_x2(n_filters, activation=self.activation, spatial_dropout=self.spatial_dropout) 
            for _ in range(n_inputs)
        ])
        
        # Excitation blocks for each branch
        self.excitation_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(n_inputs)])
        self.excitation_denses1 = nn.ModuleList([nn.Linear(n_filters, n_filters) for _ in range(n_inputs)])
        self.excitation_denses2 = nn.ModuleList([nn.Linear(n_filters, n_filters) for _ in range(n_inputs)])

    def forward(self, xs):
        outputs = []
        for x, conv, pool, dense1, dense2 in zip(
            xs, self.conv_branches, self.excitation_pools, self.excitation_denses1, self.excitation_denses2
        ):
            _conv = conv(x)
            _pool = pool(_conv).view(_conv.shape[0], -1)
            _dense1 = self.activation(dense1(_pool))
            _dense2 = torch.sigmoid(dense2(_dense1)).view(_conv.shape[0], _conv.shape[1], 1, 1)
            outputs.append(_conv * _dense2)
        y = torch.stack(outputs).sum(dim=0)
        return y

# -----------------------------------------------------------------------------------
# Parts of the ViT module
# -----------------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x

class AddClassToken(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
    
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

class AddPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
    
    def forward(self, x):
        return x + self.pos_embed

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_size, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x

class ExtractClassToken(nn.Module):
    def forward(self, x):
        return x[:, 0]

class ViTEncoder(nn.Module):
    def __init__(self, input_shape, patch_size=16, embedding_dim=768, mlp_size=3072,
                 n_blocks=12, n_heads=12, dropout=0.1):
        super().__init__()
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.patch_embedding = PatchEmbedding(input_shape[2], patch_size, embedding_dim)
        self.class_token = AddClassToken(embedding_dim)
        self.positional_embedding = AddPositionalEmbedding(num_patches, embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_dim, mlp_size, n_heads, dropout) for _ in range(n_blocks)])
        self.extract_cls = ExtractClassToken()
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.class_token(x)
        x = self.positional_embedding(x)
        x = self.blocks(x)
        x = self.extract_cls(x)
        return x

class MLP(nn.Module):
    def __init__(self, mlp_size, n_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(mlp_size, mlp_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_size, n_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class ViT(nn.Module):
    def __init__(self, input_shape, n_classes, patch_size=16, embedding_dim=768, mlp_size=3072,
                 n_blocks=12, n_heads=12, dropout=0.1):
        super().__init__()
        self.encoder = ViTEncoder(input_shape, patch_size, embedding_dim, mlp_size, n_blocks, n_heads, dropout)
        self.mlp_head = MLP(embedding_dim, n_classes, dropout)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp_head(x)
        return x

class TransResUnet(nn.Module):
    def __init__(self, input_shapes, n_outputs=9, dropout=0, inference_dropout=False):
        super().__init__()
        self.fusion = FusionBlock(
            n_filters=64, spatial_dropout=dropout, inference_dropout=inference_dropout
        )
        
        self.vit = ViTEncoder(
            input_shape=input_shapes[0],  # Assuming first input shape is the main input
            patch_size=16, 
            embedding_dim=64, 
            mlp_size=256, 
            n_blocks=12, 
            n_heads=4, 
            dropout=dropout
        )
        
        self.resunet = ResUnet(n_channels=64, n_classes=n_outputs, use_stem=False)

    def forward(self, inputs):
        fused = self.fusion(inputs)
        vit_encoded = self.vit(fused)
        vit_encoded = vit_encoded + fused  # Skip connection
        logits, _ = self.resunet(vit_encoded)
        return logits