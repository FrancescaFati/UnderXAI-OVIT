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

def main():
    parser = argparse.ArgumentParser(description='Run OV-segmentation inference on images listed in a text file.')
    parser.add_argument('txt_file', type=str, help='Path to the text file containing paths to NIfTI images.')
    parser.add_argument('--models', nargs='+', default=['pod_om'],
                        help='Name(s) of models used during inference. Options are: pod_om, abdominal_lesions, lymph_nodes. Can combine multiple.')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Increases inference speed by disabling dynamic z spacing, model ensembling and test-time augmentations.')

    args = parser.parse_args()

    txt_file = args.txt_file
    models = args.models
    fast = args.fast

    try:
        with open(txt_file, 'r') as file:
            # Strip whitespace and remove empty lines
            paths = [line.strip() for line in file.readlines() if line.strip()]

            if not paths:
                raise ValueError(f"No valid paths found in the file '{txt_file}'")

            # Process each path individually
            for single_path in paths:
                print(f"Processing: {single_path}")
                print()
                run_inference(single_path, models=models, fast=fast)
                print(f"Saving to: /home/ffati/UnderXAI-OVIT/ovseg_predictions_pod_om/{single_path}")

    except FileNotFoundError:
        print(f"Error: Could not find file '{txt_file}'")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


        