import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import albumentations as A
from typing import Optional, List, Dict
import logging
import warnings
# warnings.filterwarnings('ignore', message='.*torch.load.*')

from typing import List, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
import logging
# os.environ["NO_ALBUMENTATIONS_UPDATE"]="1" #?ALBE: Removes Albumentation warning at launch

#TODO: save all the resized pth in the bucket
#TODO: normalize 

import torch
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import torch
import os
from google.cloud import storage
from io import BytesIO

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ffati/.config/gcloud/application_default_credentials.json"
client = storage.Client(project="Underxai")
bucket = client.bucket("bucket-xai")

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from io import BytesIO
import json
from dataclasses import dataclass
from pathlib import Path

from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from io import BytesIO
import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
import nibabel as nib
from monai.data import MetaTensor
from monai.transforms import Compose, Spacing, Lambda, Resize

import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose



def windowing(image: np.ndarray, window_level: int = 40, window_width: int = 400) -> np.ndarray:
    """
    Applies windowing to a medical image, enhancing contrast within a specific intensity range.
    
    Parameters:
        image (np.ndarray): The input image.
        window_level (int): The center of the intensity window.
        window_width (int): The width of the intensity window.
    
    Returns:
        np.ndarray: The windowed image.
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / window_width  # Normalize to [0,1]
    return image


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
    
    If `return_2d_slices` is True, each __getitem__ returns a single 2D slice (with a channel dimension)
    and an annotation computed from the corresponding mask slice (0 if mask is all zeros, else 1).
    Otherwise, the full 3D volume is returned.
    """
    
    def __init__(
        self, 
        data_file: str | Path, 
        transform: Optional[Any] = None, 
        is_training: bool = True,
        excluded_years: List[int] = [2015],
        seg: bool = False,
        im_bucket: bool = False,
        seg_bucket: bool = False,
        return_2d_slices: bool = False,  # New flag for 2D slicing
        im_masked: bool = False,
        r0r1_r2: bool = False,
        contrastive: bool = False,
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
            bucket: Whether to load images from a GCS bucket.
            return_2d_slices: If True, each item is a 2D slice with annotation from the mask.
        """
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.seg = seg
        self.im_bucket = im_bucket
        self.seg_bucket = seg_bucket
        self.return_2d_slices = return_2d_slices
        self.im_masked = im_masked
        self.r0r1_r2 = r0r1_r2
        self.contrastive = contrastive
        self.seg_as_im = seg_as_im
        self.clinical_fe = clinical_fe
        
        # Initialize statistics
        self.stats = {
            'num_patients': 0,
            'residual_tumor_counts': {0: 0, 1: 0, 2: 0}
        }
        
        # Load and process patient data
        self.data = self._load_data(data_file, excluded_years)
        
        # If we need to return 2D slices, build an index mapping each slice to its patient and slice number.
        if self.return_2d_slices:
            self.slice_mapping = []
            for patient_idx, patient_data in enumerate(self.data):
                # Assuming the image tensor is shaped (C, H, W, D) with C=1.
                _, _, _, depth = patient_data.image.shape
                for slice_idx in range(depth):
                    self.slice_mapping.append((patient_idx, slice_idx))
        else:
            self.slice_mapping = None

    def _load_data(self, data_file: str | Path, excluded_years: List[int]) -> List[PatientData]:
        """Load and process the dataset."""
        data_list = []
        
        with open(data_file, 'r') as f:
            data = json.load(f)
            
            for patient in data["patients"]:
                if patient["year"] in excluded_years:
                    continue
                
                # if patient["Type of surgery"] == "IDS (interval debulking surgery)": 
                #     continue
                    
                self.stats['num_patients'] += 1
                
                # Load image: from bucket or local file
                if self.seg_as_im: 
                    image = self._load_image(patient["segmentation_path"])
                else:
                    #print(patient["Record ID"])
                    image = self._load_image_bucket(patient["image_path"]) if self.im_bucket else self._load_image(patient["image_path"])
                
                # Load mask only if seg=True
                if self.seg: 
                    mask = self._load_mask_bucket(patient["segmentation_path"]) if self.seg_bucket else self._load_mask(patient["segmentation_path"])
                else:
                    mask = None

                if self.im_masked:
                    image = image * mask 
                    mask = None      
                              
                # Process label
                original_label = patient["Residual tumor"]
                self.stats['residual_tumor_counts'][original_label] += 1
                
                label = torch.tensor(original_label, dtype=torch.long)
                # Adjust label: if label==2, convert to 1
                
                label = torch.where(label == 2, torch.tensor(1, dtype=torch.long), label)
                if self.r0r1_r2:
                    label = torch.where(label == 1, torch.tensor(0, dtype=torch.long), label)
                    
                if self.clinical_fe:
                    clinical_features = torch.tensor([
                        patient.get("Age at diagnosis", 0.0),  # Default to 0 if missing
                        #patient.get("Charlson Comorbidity Index", 0.0),
                        #patient.get("BMI (kg/m^2)", 0.0),
                        patient.get("CA125 (U/mL)", 0.0),
                        patient.get("HE4 (pmol/l)", 0.0),
                        patient.get("Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging", 0.0)
                    ], dtype=torch.float32)
                    
                    mean = clinical_features.mean(dim=0)
                    std = clinical_features.std(dim=0) + 1e-6 
                    clinical_features = (clinical_features - mean) / std
                else: 
                    clinical_features = None

                # Create PatientData instance (mask included only if seg is True)
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
    
    def _load_image_bucket(self, image_path: str) -> torch.Tensor:
        """Load image from a GCS bucket with optional caching."""
        blob = bucket.blob(image_path)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            temp_path = temp_file.name
            blob.download_to_filename(temp_path)  # Save GCS file to temp path

        # Load NIfTI image from the temporary file
        nii_img = nib.load(temp_path)

        # Get image data as a numpy array
        image_data = nii_img.get_fdata()
        img_tensor = torch.from_numpy(image_data)
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension (C, D, H, W)

        affine_matrix = torch.tensor(nii_img.affine, dtype=torch.float32)
        meta_img = MetaTensor(img_tensor, affine=affine_matrix)

        transforms = Compose([
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='trilinear'),  # Proper resampling
            Lambda(lambda x: windowing(x)),  # Apply windowing
            Resize(spatial_size=(224, 224, 256)),  # Resize to target dimensions
        ])

        # Apply transformation
        transformed_data = transforms(meta_img)

        # Clean up the temporary file
        os.remove(temp_path)
        
        return transformed_data

    def _load_mask_bucket(self, segmentation_path: str) -> torch.Tensor:
        """Load segmentation mask from a GCS bucket with optional caching."""
        blob = bucket.blob(segmentation_path)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            temp_path = temp_file.name
            blob.download_to_filename(temp_path)  # Save GCS file to temp path

        # Load NIfTI image from the temporary file
        nii_mask = nib.load(temp_path)

        # Get mask data as a numpy array
        image_data = nii_mask.get_fdata()
        mask_tensor = torch.from_numpy(image_data)
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension (C, D, H, W)

        affine_matrix = torch.tensor(nii_mask.affine, dtype=torch.float32)
        meta_mask = MetaTensor(mask_tensor, affine=affine_matrix)

        transforms = Compose([
            Spacing(pixdim=(1.0, 1.0, 1.0), mode='trilinear'),
            Resize(spatial_size=(224, 224, 256), mode='nearest'),
        ])

        transformed_data = transforms(meta_mask)
        # Normalize mask values between 0 and 1
        x_min = transformed_data.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = transformed_data.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        transformed_data = (transformed_data - x_min) / (x_max - x_min + 1e-6)

        os.remove(temp_path)
        
        return transformed_data.half()
    
    def _load_mask_bucket_pth(self, segmentation_path: str) -> torch.Tensor:
        """Load segmentation mask from a GCS bucket with optional caching."""
        blob = bucket.blob(segmentation_path)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temp_file:
            temp_path = temp_file.name
            blob.download_to_filename(temp_path)  # Save GCS file to temp path

        mask_tensor = torch.load(temp_path)

        # Normalize mask values between 0 and 1
        x_min = mask_tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x_max = mask_tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        mask_tensor = (mask_tensor - x_min) / (x_max - x_min + 1e-6)

        os.remove(temp_path)
        
        return mask_tensor
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self.return_2d_slices and self.slice_mapping is not None:
            return len(self.slice_mapping)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        If return_2d_slices is True, returns a dictionary with a 2D slice, its annotation (0/1), 
        the one-hot encoded label, and age. Otherwise, returns the full volume (with mask if available).
        """
        try:
            if self.return_2d_slices and self.slice_mapping is not None:
                # Get the mapping: which patient and which slice index.
                patient_idx, slice_idx = self.slice_mapping[idx]
                patient_data = self.data[patient_idx]
                
                # Assume patient_data.image shape: (C, D, H, W) with C=1.
                image_slice = patient_data.image[:, :, :, slice_idx]  # Resulting shape: (C, H, W)
                mask_slice = patient_data.mask[:, :, :, slice_idx]
                
                # If training and a transform is provided, apply the transform to the 2D slice.
                if self.is_training and self.transform is not None:
                    # Convert to numpy and apply the transform; note that your transform might need adjustment
                    # for 2D data if originally designed for 3D volumes.
                    transformed = self.transform(image=image_slice.squeeze(0).numpy())
                    image_slice = torch.from_numpy(transformed['image']).unsqueeze(0)
                
                # Compute annotation from the corresponding mask slice if available.
                if self.seg and patient_data.mask is not None:
                    # Using torch.all to check if the mask slice is all zero.
                    annotation = 0 if torch.all(mask_slice == 0) else 1
                else:
                    annotation = None  # or a default value if you prefer

                # One-hot encode the label (as before, converting any label==2 to 1)
                label_one_hot = F.one_hot(patient_data.label, num_classes=2).float()
                
                sample = {
                    'image': image_slice.float(),   # 2D slice
                    'annotation': annotation,         # 0 or 1 from mask slice
                    'label': label_one_hot,           # Patient-level label
                }
                
                if self.seg and patient_data.mask is not None:
                    sample['mask'] =  mask_slice.float()
                
                if self.clinical_fe: 
                    sample["clinical_features"] = patient_data.clinical_features
                
                return sample
            
            else:
                # Return the entire volume as before
                patient_data = self.data[idx]
                image = patient_data.image.float()
                
                # Apply transforms if in training mode
                if self.is_training and self.transform is not None:
                    transformed = self.transform(image=image.numpy())
                    image = torch.from_numpy(transformed['image'])
                
                # One-hot encode the label
                label_one_hot = F.one_hot(patient_data.label, num_classes=2).float()
                
                if self.contrastive: 
                    label_one_hot = patient_data.label
                
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
            print(f"Error processing item {idx}: {str(e)}")
            # Fallback to the first item if an error occurs
            return self.__getitem__(0)

    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        return self.stats



class PDSDataset(Dataset):
    def __init__(
        self, 
        data_file: str,
        transform: Optional[A.Compose] = None,
        is_training: bool = True,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.0
    ):
        self.transform = transform
        self.is_training = is_training
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        
        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.age: List[float] = []
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    if not line.strip():
                        continue
                        
                    # Split on last comma to handle paths with commas
                    parts = line.strip().rsplit(',', 1)
                    if len(parts) != 2:
                        logging.error(f"Invalid line format: {line}")
                        continue
                        
                    path, age = parts
                    path = path.strip()
                    age = float(age.strip())
                    
                    data = torch.load(path)
                    
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data)
                    if len(data.shape) == 2:
                        data = data.unsqueeze(0)
                    
                    label = int(Path(path).parts[-2])
                    if label not in [0, 1]:
                        raise ValueError(f"Invalid label {label} in {path}")
                    
                    self.data.append(data)
                    self.labels.append(label)
                    self.age.append(age)
                    
                except Exception as e:
                    logging.error(f"Error processing line: {line.strip()} - {str(e)}")
                    continue
        
        if not self.data:
            raise RuntimeError("No valid data loaded")
            
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        self.age = torch.tensor(self.age)
        
        self.class_weights = self._compute_class_weights()
    
    def _compute_class_weights(self):
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        weights = 1.0 / counts.float()
        return weights / weights.sum()
    
    # def _mixup_data(
    #     self, 
    #     x: torch.Tensor, 
    #     y: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Apply mixup augmentation"""
    #     if np.random.random() > 0.5:
    #         lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    #         batch_size = len(x)
    #         index = torch.randperm(batch_size)
            
    #         mixed_x = lam * x + (1 - lam) * x[index]
    #         mixed_y = lam * y + (1 - lam) * y[index]
    #         return mixed_x, mixed_y
    #     return x, y
    

    def _apply_label_smoothing(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing"""
        num_classes = 2
        return y * (1 - self.label_smoothing) + self.label_smoothing / num_classes
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            image = self.data[idx].float()
            label = self.labels[idx]
            age = self.age[idx]
            
            if self.is_training and self.transform is not None:
                image_np = image.numpy()
                if len(image_np.shape) == 2:
                    image_np = np.expand_dims(image_np, axis=0)
                image_np = np.transpose(image_np, (1, 2, 0))
                
                transformed = self.transform(image=image_np)
                image = torch.from_numpy(transformed['image'])
                
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)
                elif len(image.shape) == 3 and image.shape[0] != self.data[0].shape[0]:
                    image = image.permute(2, 0, 1)
            
            label_one_hot = F.one_hot(label, num_classes=2).float()
            
            if self.is_training:
                label_one_hot = self._apply_label_smoothing(label_one_hot)
            
            return {
                'image': image,
                'label': label_one_hot,
                'age': age
            }
            
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            return self[0]

class IMDataset(Dataset):
    def __init__(
        self, 
        data_file: str,
        transform: Optional[A.Compose] = None,
        is_training: bool = True,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.0
    ):
        self.transform = transform
        self.is_training = is_training
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        
        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        
        with open(data_file, 'r') as f:
            for line in f:
                data = torch.load(line.strip())
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if len(data.shape) == 2:
                    data = data.unsqueeze(0)
                    
                label = int(Path(line).parts[-2])
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label {label} in {line}")
                    
                self.data.append(data)
                self.labels.append(label)

        if not self.data:
            raise RuntimeError("No valid data loaded")
            
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            image = self.data[idx].float()
            label = self.labels[idx]
            
            if self.is_training and self.transform is not None:
                image_np = image.numpy()
                if len(image_np.shape) == 2:
                    image_np = np.expand_dims(image_np, axis=0)
                image_np = np.transpose(image_np, (1, 2, 0))
                
                transformed = self.transform(image=image_np)
                image = torch.from_numpy(transformed['image'])
                
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)
                elif len(image.shape) == 3 and image.shape[0] != self.data[0].shape[0]:
                    image = image.permute(2, 0, 1)
            
            label_one_hot = F.one_hot(label, num_classes=2).float()
            
            
            return {
                'image': image,
                'label': label_one_hot
            }
            
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            return self[0]

def get_augmentation_pipeline(self) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.2),
    ])

def read_txt_file(txt_file: str) -> List[str]:
    """
    Read paths from a text file.
    
    Args:
        txt_file: Path to the text file
        
    Returns:
        List of image paths
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(txt_file, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {txt_file}")
    except IOError as e:
        raise IOError(f"Error reading file {txt_file}: {str(e)}")