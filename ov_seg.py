import argparse
from ovseg.run.run_inference import run_inference
from monai.data import ImageDataset, DataLoader
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    Resize,
    ToTensor
)
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run OV-segmentation inference on images listed in a text file.')
    parser.add_argument('txt_file', type=str, help='Path to the text file containing paths to NIfTI images.')
    parser.add_argument('--models', nargs='+', default=['pod_om'],
                        help='Name(s) of models used during inference. Options are: pod_om, abdominal_lesions, lymph_nodes. Can combine multiple.')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Increases inference speed by disabling dynamic z spacing, model ensembling and test-time augmentations.')
    parser.add_argument('--output_mask_list_file', type=str, default='ovseg_output_masks.txt',
                        help='Path to a text file where the paths of the generated segmentation masks will be saved.')

    args = parser.parse_args()

    txt_file = args.txt_file
    models = args.models
    fast = args.fast
    output_mask_list_file = args.output_mask_list_file

    # Define the base output directory for ovseg predictions
    OVSEG_OUTPUT_BASE_DIR = Path(os.getcwd()) / "ovseg_predictions_pod_om"
    OVSEG_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(txt_file, 'r') as file:
            # Strip whitespace and remove empty lines
            paths = [line.strip() for line in file.readlines() if line.strip()]

            if not paths:
                raise ValueError(f"No valid paths found in the file '{txt_file}'")

            with open(output_mask_list_file, 'w') as output_list_f:
                # Process each path individually
                for single_path in paths:
                    print(f"Processing: {single_path}")
                    run_inference(single_path, models=models, fast=fast)

                    # Construct the expected output path for the segmentation mask
                    # ovseg mirrors the input file's relative path structure under its output directory
                    final_output_mask_path = OVSEG_OUTPUT_BASE_DIR / Path(single_path).relative_to(Path(single_path).anchor if Path(single_path).is_absolute() else '.')
                    
                    # Write the output mask path to the list file
                    output_list_f.write(str(final_output_mask_path) + '\n')
                    print(f"Recorded output mask path: {final_output_mask_path}")

    except FileNotFoundError:
        print(f"Error: Could not find file '{txt_file}'")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


        