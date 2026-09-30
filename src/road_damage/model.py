import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn

from .config import MODEL_REPO, MODEL_FILENAME


# ---------------------------------------------------------
# YOLOv12 QKV COMPATIBILITY FIX
# ---------------------------------------------------------

def compatible_aattn_forward(self, x):

    B, C, H, W = x.shape
    N = H * W

    # Loaded checkpoint uses combined QKV
    qkv = self.qkv(x).flatten(2).transpose(1, 2)

    if self.area > 1:
        qkv = qkv.reshape(
            B * self.area,
            N // self.area,
            C * 3
        )

        B, N, _ = qkv.shape

    # Split Q, K and V
    q, k, v = (
        qkv
        .view(
            B,
            N,
            self.num_heads,
            self.head_dim * 3
        )
        .permute(0, 2, 3, 1)
        .split(
            [self.head_dim, self.head_dim, self.head_dim],
            dim=2
        )
    )

    # Attention
    attn = (q.transpose(-2, -1) @ k) * (
        self.head_dim ** -0.5
    )

    # Softmax
    attn = attn.softmax(dim=-1)

    # Apply attention to Value
    x = v @ attn.transpose(-2, -1)

    x = x.permute(0, 3, 1, 2)

    v = v.permute(0, 3, 1, 2)

    if self.area > 1:

        x = x.reshape(
            B // self.area,
            N * self.area,
            C
        )

        v = v.reshape(
            B // self.area,
            N * self.area,
            C
        )

        B, N, _ = x.shape

    # Restore image dimensions
    x = (
        x
        .reshape(B, H, W, C)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    v = (
        v
        .reshape(B, H, W, C)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    # Positional encoding
    x = x + self.pe(v)

    return self.proj(x)


# Apply compatibility fix
AAttn.forward = compatible_aattn_forward


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_road_damage_model():
    """
    Download and load the YOLOv12s RDD2022 road damage model.

    Returns:
        YOLO: Loaded road damage detection model.
    """

    print("Loading road damage model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    model = YOLO(model_path)

    print("Model loaded successfully.")
    print("Classes:", model.names)

    return model