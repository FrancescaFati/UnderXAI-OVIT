from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModelForImageClassification, ViTModel, AutoModel, SamModel
from utilis import  MedViT_FeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from fromalbe import *
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ModelConfig:
    input_dim: int
    num_heads: int
    num_classes: int = 1
    max_slices: int = 256
    dropout_rate: float = 0.1
    #temperature_init: float = 0.07
    #min_temperature: float = 0.01
    adapter_reduction: int = 2
    attention_pooling: str = "contextual"
    feature_extractor: str = 'google'

class SliceProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["feature_extractor"] in ["google", "google-32", "medvit", "finetunedCT", "medsam"]:
            self.mean = [0.5,0.5,0.5]
            self.std = [0.5,0.5,0.5]
        elif config["feature_extractor"] in ["dinov2-small","dinov2-base","dino-vitb8","dino-vitb16","dino-xray"]:
            self.mean = [0.485,0.456,0.406]
            self.std = [0.229,0.224,0.225]
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.repeat(1, 3, 1, 1)  # Convert to RGB first

        # Channel-wise normalization 
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        
        x = (x - mean) / std

        return x
   
class VolumeProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["feature_extractor"] in ["google", "google-32", "medvit", "finetunedCT", "medsam"]:
            self.mean = [0.5,0.5,0.5]
            self.std = [0.5,0.5,0.5]
        elif config["feature_extractor"] in ["dinov2-small","dinov2-base","dino-vits8","dino-vitb8","dino-vitb16","dino-xray"]:
            self.mean = [0.485,0.456,0.406]
            self.std = [0.229,0.224,0.225]
            
    def forward(self, x: torch.Tensor, num_slices: int) -> torch.Tensor:
        # x shape: [BS, 1, H, W] where BS = batch_size * num_slices
        
        batch_slices = x.shape[0]
        original_batch_size = batch_slices // num_slices
        
        # Reshape to get original volumes: [B, S, H, W]
        x = x.reshape(original_batch_size, num_slices, x.shape[2], x.shape[3])
            
        # Normalize across entire volume
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-6)
        
        # Reshape back to [BS, 1, H, W]
        x = x.reshape(batch_slices, 1, x.shape[2], x.shape[3])
        # w, h = x.shape[-2:]
        # scaling_factor = 1024.0 / max(w, h)
        # new_size = [int(dim * scaling_factor) for dim in (h, w)]
        # x = F.interpolate(
        #     x,
        #     size=new_size,
        #     mode='bilinear',
        #     align_corners=True
        # )
        x = x.repeat(1, 3, 1, 1)  # Convert to RGB first

        # Channel-wise normalization 
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        
        x = (x - mean) / std

        return x
    
class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = self._choose_feature_extractor()
        self.input_dim = self._get_input_dim()
        
    def _choose_feature_extractor(self):
        
        extractors = {
            'google': ('google/vit-base-patch16-224-in21k', ViTModel),
            'google-32': ('google/vit-base-patch32-224-in21k', ViTModel),
            'dinov2-small': ('facebook/dinov2-small', AutoModel),
            'dinov2-base': ('facebook/dinov2-base', AutoModel),  
            'dino-vits8': ('facebook/dino-vits8', AutoModel),                      
            'dino-xray': ('StanfordAIMI/dinov2-base-xray-224',AutoModel),
            'dino-vitb8': ('facebook/dino-vitb8', ViTModel),
            'dino-vitb16': ('facebook/dino-vitb16', ViTModel),
            'finetunedCT': ('Manuel-O/vit-base-patch16-224-in21k-finetuned-CT', AutoModelForImageClassification),
            'medvit': (None, MedViT_FeatureExtractor),
            'medsam':('flaviagiammarino/medsam-vit-base', SamModel)
        }
        
        model_info = extractors.get(self.config['feature_extractor'])
        if not model_info:
            raise ValueError(f"Invalid feature extractor: {self.config['feature_extractor']}")
            
        path, model_class = model_info
        if path is None:  # Special case for MedViT
            return model_class().to(device)
        elif model_class == AutoModelForImageClassification:
            return model_class.from_pretrained(path).vit.to(device)
        elif model_class == SamModel:
            return model_class.from_pretrained(path).vision_encoder.to(device)
        else:
            return model_class.from_pretrained(path).to(device)
            
    def _get_input_dim(self):
        if self.config['feature_extractor'] == 'medvit': 
            input_dim = 1024
        elif self.config['feature_extractor'] == 'medsam': 
            input_dim = 256 
        else:
            input_dim = self.feature_extractor.config.hidden_size
        return input_dim
    
    def get_extractor(self):
        return self.feature_extractor
    
    def get_input_dim(self):
        return self.input_dim 
    
class ConvAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # 256 -> 128
            nn.Conv1d(input_dim, input_dim, 5, stride=2, padding=2),
            nn.GELU(),
            # 128 -> 64 
            nn.Conv1d(input_dim, input_dim, 5, stride=2, padding=2),
            nn.GELU(),
            # 64 -> 32
            nn.Conv1d(input_dim, input_dim, 7, stride=2, padding=3),
            nn.GELU(),
            # 32 -> 8
            nn.Conv1d(input_dim, input_dim, 7, stride=4, padding=3),
            nn.GELU(),
            # 8 -> 1
            nn.Conv1d(input_dim, input_dim, 8, stride=1),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)  # Output: [B, E, 1]
        x = x.squeeze(-1)        # [B, E]
        
        # dummy_weights = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)
        return x, None


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, dropout_rate, attention_resc=1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim//attention_resc)
        self.key = nn.Linear(input_dim, input_dim//attention_resc)
        self.value = nn.Linear(input_dim, input_dim//attention_resc)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, num_slices, input_dim]
        x = self.norm(x)
        
        # Compute Q, K, V
        Q = self.query(x)  # [B, S, D]
        K = self.key(x)    # [B, S, D]
        V = self.value(x)  # [B, S, D]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))  # [B, S, S]
        weights = self.drop(F.softmax(scores, dim=-1))  # [B, S, S]
        
        # Compute weighted sum
        attended = torch.matmul(weights, V)  # [B, S, D]
        
        # Pool across slices
        pooled = attended.mean(dim=1)  # [B, D]
        
        return pooled, weights

class CrossAttentionPooling(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_rate, attention_resc=1):
        super().__init__()
        # Linear projections for the query, key, and value
        self.query_proj = nn.Linear(query_dim, query_dim // attention_resc)
        self.key_proj = nn.Linear(key_dim, key_dim // attention_resc)
        self.value_proj = nn.Linear(value_dim, value_dim // attention_resc)
        # Layer normalization for each input type
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_key = nn.LayerNorm(key_dim)
        self.norm_value = nn.LayerNorm(value_dim)        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query_matrix: torch.Tensor, key_matrix: torch.Tensor, value_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_matrix: Tensor of shape [B, Q_len, query_dim]
            key_value_matrix: Tensor of shape [B, KV_len, key_value_dim]
            
        Returns:
            pooled: Tensor of shape [B, proj_dim] (pooled output)
            weights: Attention weights of shape [B, Q_len, KV_len]
        """
        # Normalize the inputs
        query_matrix = self.norm_query(query_matrix)
        key_matrix = self.norm_key(key_matrix)
        value_matrix = self.norm_value(value_matrix)
        
        # Compute projections for query, key, and value
        Q = self.query_proj(query_matrix)       # [B, Q_len, proj_dim]
        K = self.key_proj(key_matrix)       # [B, KV_len, proj_dim]
        V = self.value_proj(value_matrix)     # [B, KV_len, proj_dim]
        
        # Compute attention scores between the query and keys
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))  # [B, Q_len, KV_len]
        weights = self.dropout(F.softmax(scores, dim=-1))  # [B, Q_len, KV_len]
        
        # Compute the weighted sum of values
        attended = torch.matmul(weights, V)  # [B, Q_len, proj_dim]
        
        # Pool the attended features across the query length (e.g., via mean pooling)
        pooled = attended.mean(dim=1)  # [B, proj_dim]
        
        return pooled, weights


class ProjectionHead(torch.nn.Module):
    def __init__(self, feature_dim, projection_dim=128):
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, projection_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)  # Normalizzazione L2


class Slice_Contrastive(nn.Module):
    """
    Main classifier module for contrastive learning at slice level.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.slice_processor = SliceProcessor(config)
        
        self.feature_extractor =  FeatureExtractor(config).get_extractor()
        self.input_dim = FeatureExtractor(config).get_input_dim()
        
        self.projection_head = ProjectionHead(self.input_dim)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            tuple: (embeddings)
        """
 
        processed_slices = self.slice_processor(images)     # (B,3,H,W)
        
        # Extract features 
        with torch.no_grad():
            embeddings = self.feature_extractor(processed_slices).last_hidden_state[:, 0, :]
        
        projected_embeddings = self.projection_head(embeddings)
        
        return projected_embeddings

class Volume_Contrastive(nn.Module):
    """
    Main classifier module for Contrastive Learning at volume level.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        
        self.slice_processor = VolumeProcessor(config)
        self.feature_extractor =  FeatureExtractor(config).get_extractor()
        self.input_dim = FeatureExtractor(config).get_input_dim()
        self.num_classes = 2
        self.drodropout_rate = 0
        self.num_heads = 2
        self.attention_pooling = 'contextual'
        self.max_slices = 256
        self.attention_resc = 2
        
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Processors
        
        self.attention_pool = nn.ModuleDict({
            'contextual': AttentionPooling(self.input_dim, self.drodropout_rate, self.attention_resc),
            'conv':ConvAttentionPooling(self.input_dim),
        })[self.config.attention_pooling]

        self.projection_head = ProjectionHead(self.input_dim//self.attention_resc)

        
    def _validate_input(self, images: torch.Tensor):
        """Validates input tensor dimensions and size."""
        if images.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,H,W,S), got {images.dim()}D")
        if images.size(1) != 1:  
            raise ValueError(f"Expected 1 channel, got {images.size(1)}") 
        if images.size(-1) > self.max_slices:
            raise ValueError(f"Max slices exceeded: {images.size(-1)} > {self.max_slices}")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input tensor of shape [batch_size, channels, height, width, slices]
            
        Returns:
            tuple: (embeddings)
        """
        # self._validate_input(images)
        
        extra = {}
        batch_size, _, height, width, num_slices = images.shape
        
        # Process slices
        slices = images.permute(0, 4, 1, 2, 3)                          # (B,S,1,H,W) 
        extra["s"]=slices
        slices = slices.reshape(-1, 1, height, width)                   # BS,1,H,W)
        processed_slices = self.slice_processor(slices, num_slices)     # (BS,3,H,W)
        extra["ps"]=processed_slices
        
        # Extract features 
        with torch.no_grad():
            features = self.feature_extractor(processed_slices).last_hidden_state[:, 0, :]  # (BS,E)
        
        extra["f1"] = features
        
        # Reshape and add positional encoding
        features = features.reshape(batch_size, num_slices, -1)  # (B,S,E)
        extra["f2"] = features
        
        #features = features + self.pos_encoding[:, :num_slices]
        
        # Apply attention pooling
        pooled_features, attention_weights = self.attention_pool(features) 
        extra["p"] = pooled_features
        extra["a"] = attention_weights
        
        projected_features = self.projection_head(pooled_features)
        
        return projected_features


class ClinicalDataEncoder(nn.Module):
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
        return self.encoder(x)

    def get_output_dim(self):
        """Returns the output feature dimension after encoding."""
        return self.encoding_dim




class Vit_Classifier(nn.Module):
    """
    Main classifier module for processing volumetric medical images using ViT and attention pooling.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.feature_extractor =  FeatureExtractor(config).get_extractor()
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
        self.drodropout_rate =config['dropout']
        self.num_heads = config['num_heads']
        self.attention_pooling = config['attention_pooling']
        self.max_slices = 256
        self.attention_resc = config['attention_resc']
        self.clinical_features = config["clinical_features"]
        
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Processors
        self.volume_processor = VolumeProcessor(config)
        
        self.attention_pool = nn.ModuleDict({
            'contextual': AttentionPooling(self.input_dim, self.drodropout_rate, self.attention_resc),
            'conv':ConvAttentionPooling(self.input_dim),
            'cross': CrossAttentionPooling(self.input_dim, self.input_dim, self.input_dim, self.drodropout_rate, self.attention_resc)
        })[self.attention_pooling]
    
    
        ct_feature_dim = self.input_dim // self.attention_resc
        combined_dim = ct_feature_dim + clinical_encoded_dim if self.clinical_features_enabled else ct_feature_dim
        combined_dim = combined_dim + ct_feature_dim if self.config['mask_att'] else combined_dim
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(combined_dim),
        #     nn.Linear(combined_dim, combined_dim // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(combined_dim // 2),
        #     nn.Linear(combined_dim // 2, self.num_classes)
        # )    
    
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),  # Normalize input features
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),  # Normalization before activation
            nn.ReLU(),
            #nn.Dropout(0.1),  # Dropout for regularization
            nn.Linear(combined_dim // 2, combined_dim // 8),
            nn.BatchNorm1d(combined_dim // 8),
            nn.ReLU(),
            #nn.Dropout(0.1),
            # nn.Linear(combined_dim // 4, combined_dim // 8),
            # nn.BatchNorm1d(combined_dim // 8),
            # nn.ReLU(),

            # nn.Linear(combined_dim // 8, combined_dim // 16),
            # nn.BatchNorm1d(combined_dim // 16),
            # nn.ReLU(),

            nn.Linear(combined_dim // 8, combined_dim // 32),
            nn.BatchNorm1d(combined_dim // 32),
            nn.ReLU(),

            nn.Linear(combined_dim // 32, self.num_classes)  # Final output layer
        )
    
        
    def _validate_input(self, images: torch.Tensor):
        """Validates input tensor dimensions and size."""
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
            
        Returns:
            tuple: (logits, attention_weights)
        """
        # self._validate_input(images)
        
        extra = {}
        #images = images.unsqueeze(1)
        batch_size, _, height, width, num_slices = images.shape
        
        # Process slices
        slices = images.permute(0, 4, 1, 2, 3)                          # (B,S,1,H,W) 
        extra["s"]=slices
        slices = slices.reshape(-1, 1, height, width)                   # BS,1,H,W)
        processed_slices = self.volume_processor(slices, num_slices)     # (BS,3,H,W)
        extra["ps"]=processed_slices
            
        # Extract features 
        with torch.no_grad():
            if self.config['feature_extractor'] == 'medvit':
                features = self.feature_extractor(processed_slices)
            elif self.config['feature_extractor'] == 'medsam': 
                features = self.feature_extractor(processed_slices).last_hidden_state
            else:
                features = self.feature_extractor(processed_slices).last_hidden_state[:, 0, :]  # (BS,E)
        extra["f1"] = features
        
        # if self.config['mask']: 
        #     mask_slices = masks.permute(0, 4, 1, 2, 3)                           
        #     mask_slices = mask_slices.reshape(-1, 1, height, width)                   
        #     mask_processed_slices = self.volume_processor(mask_slices, num_slices)     
            
        #     with torch.no_grad():
        #         if self.config['feature_extractor'] == 'medvit':
        #             mask_features = self.feature_extractor(mask_processed_slices)
        #         elif self.config['feature_extractor'] == 'medsam': 
        #             mask_features = self.feature_extractor(mask_processed_slices).last_hidden_state
        #         else:
        #             mask_features = self.feature_extractor(mask_processed_slices).last_hidden_state[:, 0, :]  # (BS,E)
        #     mask_features = features.reshape(batch_size, num_slices, -1)
        # else: 
        #     mask_features = None
        
        # Reshape and add positional encoding
        features = features.reshape(batch_size, num_slices, -1)  # (B,S,E)
        extra["f2"] = features
        
        #features = features + self.pos_encoding[:, :num_slices]
        
        # if self.config['attention_pooling'] == 'cross': 
        #     if self.config["query"] == 'mask':
        #         pooled_features, attention_weights = self.attention_pool(mask_features, features, features) 
        #     elif self.config["query"] == 'im':
        #         pooled_features, attention_weights = self.attention_pool(features, mask_features, features) 
        # # Apply attention pooling
        # else: 
        #     pooled_features, attention_weights = self.attention_pool(features) 
        #     if self.config['mask_att']: 
        #         mask_pooled_features, _ = self.attention_pool(mask_features)
        #     else: 
        #         mask_pooled_features = None
        
        pooled_features, attention_weights = self.attention_pool(features) 
              
        extra["p"] = pooled_features
        extra["a"] = attention_weights
        # pooled_features -> (B,E) or (B, E//att_res)
        # attn_weights    -> (B,S,S)
        
        # if self.config['mask_att']: 
        #     pooled_features = torch.cat([pooled_features, mask_pooled_features], dim=1)
        # else: 
        #     pooled_features = pooled_features
        
        
        #pooled_features = self.adapter(pooled_features) 
        if self.clinical_features_enabled:
            clinical_encoded = self.clinical_encoder(clinical_features)
            combined_features = torch.cat([pooled_features, clinical_encoded], dim=1)
            logits = self.classifier(combined_features)
        else:
            logits = self.classifier(pooled_features)
            
        #threshold = self.threshold_net(age.unsqueeze(1))
        #adjusted_pred = (torch.sigmoid(logits) > threshold).float()
        
        return logits, attention_weights, pooled_features 











