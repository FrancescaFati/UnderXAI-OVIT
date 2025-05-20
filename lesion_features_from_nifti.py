import argparse
import logging
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage
from skimage.measure import marching_cubes

# TODO: Set up logging as needed
logging.basicConfig(level=logging.INFO)

def compute_total_lesion_features(nii, lesion_values=[1, 9]):
    """
    Computes total volume, surface area, compactness, fractal dimension, and number of sub-lesions for each lesion type.

    Parameters:
        nii (nibabel object): Loaded NIfTI image object.
        lesion_values (list): List of lesion intensity values to analyze.

    Returns:
        dict: Dictionary containing total volume (mm³), surface area (mm²), compactness, fractal dimension, and number of sub-lesions.
    """
    data = nii.get_fdata()
    voxel_spacing = nii.header.get_zooms()  # dx, dy, dz in mm
    voxel_volume_mm3 = np.prod(voxel_spacing)

    lesion_totals = {}

    for lesion_value in lesion_values:
        lesion_mask = (data == lesion_value).astype(np.uint8)
        lesion_voxel_count = np.sum(lesion_mask)
        total_volume_mm3 = lesion_voxel_count * voxel_volume_mm3

        if np.any(lesion_mask):
            # Marching Cubes with real-world spacing
            verts, faces, _, _ = marching_cubes(lesion_mask, level=0, spacing=voxel_spacing)

            # Compute triangle areas
            tris = verts[faces]
            vec1 = tris[:, 1] - tris[:, 0]
            vec2 = tris[:, 2] - tris[:, 0]
            cross_prod = np.cross(vec1, vec2)
            area = 0.5 * np.linalg.norm(cross_prod, axis=1)
            surface_area_mm2 = np.sum(area)
        else:
            surface_area_mm2 = 0

        compactness = (surface_area_mm2 ** 2) / total_volume_mm3 if total_volume_mm3 > 0 else 0

        def fractal_dimension(image, threshold=0.5):
            assert len(image.shape) == 3
            def box_count(Z, k):
                S = np.add.reduceat(np.add.reduceat(np.add.reduceat(Z,
                        np.arange(0, Z.shape[0], k), axis=0),
                        np.arange(0, Z.shape[1], k), axis=1),
                        np.arange(0, Z.shape[2], k), axis=2)
                return len(np.where((S > 0) & (S < k ** 3))[0])

            Z = image > threshold
            sizes = 2 ** np.arange(1, int(np.log2(min(Z.shape))) + 1)
            counts = np.array([box_count(Z, size) for size in sizes])
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -coeffs[0]

        fractal_dim = fractal_dimension(lesion_mask) if np.any(lesion_mask) else 0
        _, num_sublesions = scipy.ndimage.label(lesion_mask)

        lesion_totals[lesion_value] = {
            "Number of Sub-lesions": num_sublesions,
            "Total Volume (mm³)": total_volume_mm3,
            "Total Surface Area (mm²)": surface_area_mm2,
            "Compactness": compactness,
            "Fractal Dimension": fractal_dim,
        }

    return lesion_totals

def main():
    parser = argparse.ArgumentParser(description="Compute lesion features from a NIfTI segmentation file.")
    parser.add_argument('--nifti', required=True, help='Path to the local NIfTI file')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    args = parser.parse_args()

    seg_nii = nib.load(args.nifti)
    lesion_features = compute_total_lesion_features(seg_nii, lesion_values=[1, 9])

    results = []
    for lesion_type, features in lesion_features.items():
        results.append({
            "Lesion Type": lesion_type,
            "Number of Sub-lesions": features["Number of Sub-lesions"],
            "Total Volume (mm³)": features["Total Volume (mm³)"],
            "Total Surface Area (mm²)": features["Total Surface Area (mm²)"],
            "Compactness": features["Compactness"],
            "Fractal Dimension": features["Fractal Dimension"]
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()