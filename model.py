from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModelForImageClassification, ViTModel, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ModelConfig:
    """
    Configuration dataclass for model hyperparameters and settings.
    """
    input_dim: int
    num_heads: int
    num_classes: int = 1
    max_slices: int = 256
    dropout_rate: float = 0.1
    feature_extractor: str = 'google-32'


class VolumeProcessor(nn.Module):
    """
    Preprocesses volumetric medical images for feature extraction.
    Handles normalization and channel conversion for input slices.
    """
    def __init__(self, config):
        super().__init__()
        # Set normalization parameters based on feature extractor type
        if config["feature_extractor"] in ["google-16", "google-32"]:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif config["feature_extractor"] in ["dinov2-small", "dinov2-base"]:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def forward(self, x: torch.Tensor, num_slices: int) -> torch.Tensor:
        """
        Normalize and convert grayscale slices to RGB for transformer input.
        Args:
            x: Input tensor of shape [batch_size * num_slices, 1, H, W]
            num_slices: Number of slices per volume
        Returns:
            Normalized tensor of shape [batch_size * num_slices, 3, H, W]
        """
        batch_slices = x.shape[0]
        original_batch_size = batch_slices // num_slices

        # Reshape to [B, S, H, W] to process as volumes
        x = x.reshape(original_batch_size, num_slices, x.shape[2], x.shape[3])

        # Normalize across the entire volume
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-6)

        # Reshape back to [BS, 1, H, W]
        x = x.reshape(batch_slices, 1, x.shape[2], x.shape[3])
        x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels (RGB)

        # Channel-wise normalization
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

class FeatureExtractor:
    """
    Wrapper for selecting and loading a transformer-based feature extractor.
    """
    def __init__(self, config):
        self.config = config
        self.feature_extractor = self._choose_feature_extractor()
        self.input_dim = self._get_input_dim()

    def _choose_feature_extractor(self):
        """
        Select and load the appropriate transformer model for feature extraction.
        Returns:
            Pretrained transformer model
        """
        extractors = {
            'google-16': ('google/vit-base-patch16-224-in21k', ViTModel),
            'google-32': ('google/vit-base-patch32-224-in21k', ViTModel),
            'dinov2-small': ('facebook/dinov2-small', AutoModel),
            'dinov2-base': ('facebook/dinov2-base', AutoModel),
        }
        model_info = extractors.get(self.config['feature_extractor'])
        if not model_info:
            raise ValueError(f"Invalid feature extractor: {self.config['feature_extractor']}")
        path, model_class = model_info
        if model_class == AutoModelForImageClassification:
            return model_class.from_pretrained(path).vit.to(device)
        else:
            return model_class.from_pretrained(path).to(device)

    def _get_input_dim(self):
        """
        Get the hidden size (feature dimension) of the transformer model.
        Returns:
            int: Hidden size
        """
        input_dim = self.feature_extractor.config.hidden_size
        return input_dim

    def get_extractor(self):
        return self.feature_extractor

    def get_input_dim(self):
        return self.input_dim

class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling across slices of a volume.
    """
    def __init__(self, input_dim, dropout_rate, attention_resc=1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim // attention_resc)
        self.key = nn.Linear(input_dim, input_dim // attention_resc)
        self.value = nn.Linear(input_dim, input_dim // attention_resc)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-pooled features across slices.
        Args:
            x: Tensor of shape [batch_size, num_slices, input_dim]
        Returns:
            pooled: Tensor of shape [batch_size, pooled_dim]
            weights: Attention weights of shape [batch_size, num_slices, num_slices]
        """
        x = self.norm(x)
        Q = self.query(x)  # [B, S, D]
        K = self.key(x)    # [B, S, D]
        V = self.value(x)  # [B, S, D]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))  # [B, S, S]
        weights = self.drop(F.softmax(scores, dim=-1))  # [B, S, S]
        attended = torch.matmul(weights, V)  # [B, S, D]
        pooled = attended.mean(dim=1)  # [B, D]
        return pooled, weights

class ClinicalDataEncoder(nn.Module):
    """
    Encodes tabular clinical data into a learned feature representation.
    """
    def __init__(self, clinical_dim, encoding_dim):
        super().__init__()
        self.clinical_dim = clinical_dim
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass for clinical data encoding.
        Args:
            x: Input tensor of shape [batch_size, clinical_dim]
        Returns:
            Encoded tensor of shape [batch_size, encoding_dim]
        """
        return self.encoder(x)

    def get_output_dim(self):
        """
        Returns the output feature dimension after encoding.
        """
        return self.encoding_dim

class Vit_Classifier(nn.Module):
    """
    Main classifier module for volumetric medical images using ViT and attention pooling.
    Optionally incorporates clinical tabular data.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config).get_extractor()
        self.input_dim = FeatureExtractor(config).get_input_dim()
        self.clinical_features_enabled = config["clinical_features"]
        if self.clinical_features_enabled:
            clinical_input_dim = config["clinical_input_dim"]
            clinical_encoding_dim = config["clinical_encoding_dim"]
            self.clinical_encoder = ClinicalDataEncoder(clinical_input_dim, clinical_encoding_dim)
            clinical_encoded_dim = self.clinical_encoder.get_output_dim()
        else:
            self.clinical_encoder = None
            clinical_encoded_dim = 0
        self.num_classes = config['num_classes']
        self.drodropout_rate = config['dropout']
        self.num_heads = config['num_heads']
        self.attention_pooling = config['attention_pooling']
        self.max_slices = 256
        self.attention_resc = config['attention_resc']
        self.clinical_features = config["clinical_features"]
        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Image volume processor
        self.volume_processor = VolumeProcessor(config)
        self.attention_pool = AttentionPooling(self.input_dim, self.drodropout_rate, self.attention_resc)
        ct_feature_dim = self.input_dim // self.attention_resc
        combined_dim = ct_feature_dim + clinical_encoded_dim if self.clinical_features_enabled else ct_feature_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, combined_dim // 8),
            nn.BatchNorm1d(combined_dim // 8),
            nn.ReLU(),
            nn.Linear(combined_dim // 8, combined_dim // 32),
            nn.BatchNorm1d(combined_dim // 32),
            nn.ReLU(),
            nn.Linear(combined_dim // 32, self.num_classes)
        )

    def _validate_input(self, images: torch.Tensor):
        """
        Validates input tensor dimensions and size for volumetric images.
        Args:
            images: Input tensor of shape [B, C, H, W, S]
        Raises:
            ValueError: If input does not match expected dimensions or exceeds max slices.
        """
        if images.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,H,W,S), got {images.dim()}D")
        if images.size(1) != 1:
            raise ValueError(f"Expected 1 channel, got {images.size(1)}")
        if images.size(-1) > self.max_slices:
            raise ValueError(f"Max slices exceeded: {images.size(-1)} > {self.max_slices}")

    def forward(self, images: torch.Tensor, clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        Args:
            images: Input tensor of shape [batch_size, channels, height, width, slices]
            clinical_features: Optional clinical data tensor [batch_size, clinical_dim]
        Returns:
            tuple: (logits, attention_weights, pooled_features)
        """
        batch_size, _, height, width, num_slices = images.shape
        # Rearrange and flatten slices for feature extraction
        slices = images.permute(0, 4, 1, 2, 3)  # (B, S, 1, H, W)
        slices = slices.reshape(-1, 1, height, width)  # (B*S, 1, H, W)
        processed_slices = self.volume_processor(slices, num_slices)  # (B*S, 3, H, W)
        # Extract features using transformer backbone
        with torch.no_grad():
            features = self.feature_extractor(processed_slices).last_hidden_state[:, 0, :]  # (B*S, E)
        features = features.reshape(batch_size, num_slices, -1)  # (B, S, E)
        pooled_features, attention_weights = self.attention_pool(features)
        if self.clinical_features_enabled:
            clinical_encoded = self.clinical_encoder(clinical_features)
            combined_features = torch.cat([pooled_features, clinical_encoded], dim=1)
            logits = self.classifier(combined_features)
        else:
            logits = self.classifier(pooled_features)
        return logits, attention_weights, pooled_features











