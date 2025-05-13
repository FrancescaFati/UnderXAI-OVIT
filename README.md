# OC_seg_analysis

This script computes lesion features (volume, surface area, compactness, fractal dimension, and number of sub-lesions) from a local NIfTI segmentation file. Results are saved as a CSV file.

## Requirements
- Python 3.7+
- See `requirements.txt` for Python dependencies

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script with the required arguments:
```bash
python lesion_features_from_nifti.py --nifti /path/to/your_segmentation.nii.gz --output /path/to/output.csv
```

- `--nifti`: Path to the local NIfTI segmentation file
- `--output`: Path to the output CSV file

## Notes
- The script processes a single NIfTI file and writes the computed lesion features to the specified CSV.
- The script expects the NIfTI file to use the same lesion value conventions as in the code (default: 1 and 9).

## License
[MIT](LICENSE)