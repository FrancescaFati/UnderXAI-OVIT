"""
PCS Dataset utilities for PyTorch.
This module provides a PyTorch Dataset class and related utilities for loading and processing PCS (PRIMARY-CYTOREDUCTIVE SURGERY) data from local files.

"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import logging
import json
from dataclasses import dataclass


@dataclass
class PatientData:
    """Data class to store patient information."""
    image: torch.Tensor  # Expected shape: (C, H, W, D) where C=1
    label: torch.Tensor
    clinical_features: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None  # Mask is optional

class PCSDataset(Dataset):
    """
    Dataset class for PCS (PRIMARY-CYTOREDUCTIVE SURGERY) data.
    Loads patient data from a JSON file and local image/mask files.
    """
    def __init__(
        self, 
        data_file: str | Path, 
        transform: Optional[Any] = None, 
        is_training: bool = True,
        excluded_years: List[int] = [2015],
        seg: bool = False,
        seg_as_im: bool = False,
        clinical_fe: bool = False, 
    ):
        """
        Initialize the PCS Dataset.
        Args:
            data_file: Path to the JSON data file.
            transform: Optional transform to be applied to images.
            is_training: Whether this dataset is for training.
            excluded_years: List of years to exclude from the dataset.
            seg: Whether to include segmentation masks.
            seg_as_im: Use segmentation as image input.
            clinical_fe: Include clinical features.
        """
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.seg = seg
        self.seg_as_im = seg_as_im
        self.clinical_fe = clinical_fe
        self.stats = {
            'num_patients': 0,
            'residual_tumor_counts': {0: 0, 1: 0, 2: 0}
        }
        self.data = self._load_data(data_file, excluded_years)

    def _load_data(self, data_file: str | Path, excluded_years: List[int]) -> List[PatientData]:
        """Load and process the dataset from a JSON file."""
        data_list = []
        with open(data_file, 'r') as f:
            data = json.load(f)
            for patient in data["patients"]:
                if patient["year"] in excluded_years:
                    continue
                self.stats['num_patients'] += 1
                if self.seg_as_im: 
                    image = self._load_image(patient["segmentation_path"])
                else:
                    image = self._load_image(patient["image_path"])
                if self.seg: 
                    mask = self._load_mask(patient["segmentation_path"])
                else:
                    mask = None
                original_label = patient["Residual tumor"]
                self.stats['residual_tumor_counts'][original_label] += 1
                label = torch.tensor(original_label, dtype=torch.long)
                label = torch.where(label == 2, torch.tensor(1, dtype=torch.long), label)
                if self.clinical_fe:
                    clinical_features = torch.tensor([
                        patient.get("Age at diagnosis", 0.0),
                        patient.get("CA125 (U/mL)", 0.0),
                        patient.get("HE4 (pmol/l)", 0.0),
                        patient.get("Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging", 0.0)
                    ], dtype=torch.float32)
                    mean = clinical_features.mean(dim=0)
                    std = clinical_features.std(dim=0) + 1e-6 
                    clinical_features = (clinical_features - mean) / std
                else: 
                    clinical_features = None
                patient_data = PatientData(image=image, label=label, clinical_features=clinical_features, mask=mask)
                data_list.append(patient_data)
        return data_list

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image from a given path (local file)."""
        return torch.load(image_path)

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load segmentation mask from a given path (local file)."""
        x = torch.load(mask_path)
        x_min = x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return x
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        Returns a dictionary with image, label, and optionally mask and clinical features.
        """
        try:
            patient_data = self.data[idx]
            image = patient_data.image.float()
            if self.is_training and self.transform is not None:
                transformed = self.transform(image=image.numpy())
                image = torch.from_numpy(transformed['image'])
            label_one_hot = F.one_hot(patient_data.label, num_classes=2).float()
            sample = {
                'image': image,
                'label': label_one_hot,
            }
            if self.seg and patient_data.mask is not None:
                sample['mask'] = patient_data.mask.float()
            if self.clinical_fe: 
                sample['clinical_features'] = patient_data.clinical_features
            return sample
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Fallback to the first item if an error occurs
            return self.__getitem__(0)

    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        return self.stats

# No main guard or script logic; this is a pure module for import and reuse.


