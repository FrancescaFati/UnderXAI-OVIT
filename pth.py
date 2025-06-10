#!/usr/bin/env python
# coding: utf-8

# In[11]:


from monai.data import ImageDataset, DataLoader

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    Resize,
    ToTensor, 
    Spacing
)

from monai.transforms import Lambda
import numpy as np
import torch
import nibabel as nib
import os
from pathlib import Path
from monai.data import MetaTensor
import argparse
import re # Import regex module

def windowing(image: np.ndarray, window_level: int = 40, window_width: int = 400) -> np.ndarray:
    """
    Applies windowing to a medical image, enhancing contrast within a specific intensity range.

    Parameters:
    - image (np.ndarray): The input image.
    - window_level (int): The center of the intensity window.
    - window_width (int): The width of the intensity window.

    Returns:
    - np.ndarray: The windowed image.
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / window_width
    return image

def open_txt_file(txt_file: str):
    im_paths = []
    with open(txt_file, 'r') as file:
        for line in file:
            # Remove newline characters and any trailing spaces
            line = line.strip()
            im_paths.append(line)
    return im_paths

def process_images(file_list, transformations_im, transformations_mask, output_base_dir):
    for file_path in file_list:
        
        is_prediction_mask = 'ovseg_predictions_pod_om' in file_path
        
        # Construct output directory
        output_dir = Path(output_base_dir) / "pth" 
        output_dir.mkdir(parents=True, exist_ok=True)
                
        nifti_img = nib.load(file_path)
        img_data = torch.from_numpy(nifti_img.get_fdata())
        img_data = img_data.unsqueeze(0)  # Add channel dim first
        
        affine_matrix = torch.tensor(nifti_img.affine, dtype=torch.float32)
        meta_img = MetaTensor(img_data, affine=affine_matrix)
        
        # Apply transforms based on file type
        if is_prediction_mask: 
            transformed_data = transformations_mask(meta_img)
            output_file_name = "OC_mask.pth"
            print(f"Saving prediction mask to: {output_dir / output_file_name}")
        else: 
            transformed_data = transformations_im(meta_img)
            output_file_name = "CT.pth"
            print(f"Saving CT image to: {output_dir / output_file_name}")
            
        torch.save(transformed_data, output_dir / output_file_name)

transformations_im = Compose([#EnsureChannelFirst(), 
                              Spacing(pixdim=(1.0, 1.0, 1.0), mode='trilinear'), 
                              Lambda(lambda x: windowing(x)),
                              Resize(spatial_size=(224,224,256)),
                              ])

transformations_mask = Compose([#EnsureChannelFirst(), 
                              Spacing(pixdim=(1.0, 1.0, 1.0), mode='trilinear'), 
                              Resize(spatial_size=(224,224,256), mode = "nearest"),
                              ToTensor()])

def main():
    parser = argparse.ArgumentParser(description='Process NIfTI images to PTH format with MONAI transforms.')
    parser.add_argument('input_txt_file', type=str, 
                        help='Path to the text file containing paths to NIfTI images.')
    parser.add_argument('--output_dir', type=str, default='./processed_pth_data',
                        help='Base directory to save the processed PTH files. '\
                             'Files will be organized under <output_dir>/<year>/pth/<id_name>/.')
    
    args = parser.parse_args()

    im_paths = open_txt_file(args.input_txt_file)

    process_images(im_paths, transformations_im, transformations_mask, args.output_dir)

if __name__ == '__main__':
    main()






