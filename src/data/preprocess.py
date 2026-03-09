import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess FITS images to RGB.")
    parser.add_argument("--raw_dir", type=str, default="../data/raw/", help="Directory containing raw FITS")
    parser.add_argument("--processed_dir", type=str, default="../data/processed/rgb_images/", help="Directory to save RGB images")
    parser.add_argument("--Q", type=float, default=8, help="Asinh softening parameter for Lupton RGB")
    parser.add_argument("--stretch", type=float, default=0.5, help="Linear stretch parameter for Lupton RGB")
    return parser.parse_args()

def safe_read_fits(file_path):
    """Safely open and read data from a FITS file."""
    if not os.path.exists(file_path):
        return None
    try:
        with fits.open(file_path) as hdul:
            # Data is usually in the primary HDU for simple images
            data = hdul[0].data
            if data is None:
                # Fallback to first extension if primary is empty header
                 data = hdul[1].data
            # Deal with NaNs or infs from instrument failures
            data = np.nan_to_num(data)
            return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def make_rgb_components(raw_dir, objid):
    """
    Combines SDSS i, r, g bands to create an RGB image.
    SDSS typically maps Filter i -> Red, Filter r -> Green, Filter g -> Blue.
    """
    i_path = os.path.join(raw_dir, 'i', f"{objid}_i.fits")
    r_path = os.path.join(raw_dir, 'r', f"{objid}_r.fits")
    g_path = os.path.join(raw_dir, 'g', f"{objid}_g.fits")
    
    i_data = safe_read_fits(i_path)
    r_data = safe_read_fits(r_path)
    g_data = safe_read_fits(g_path)
    
    if i_data is None or r_data is None or g_data is None:
        return None, None, None
        
    return i_data, r_data, g_data

def process_fits_to_rgb(raw_dir, processed_dir, Q=8, stretch=0.5):
    """
    Pipeline to find all downloaded FITS in the raw directory and convert
    them to high-quality RGB images suitable for CNNs using Lupton mapping.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Identify unique OBJIDs from whatever is in the 'g' band folder
    target_band_dir = os.path.join(raw_dir, 'g')
    if not os.path.exists(target_band_dir):
        print(f"Directory {target_band_dir} not found. Have you downloaded data yet?")
        return
        
    fits_files = [f for f in os.listdir(target_band_dir) if f.endswith('_g.fits')]
    objids = [f.split('_')[0] for f in fits_files]
    
    print(f"Found {len(objids)} objects to process.")
    
    processed_count = 0
    for objid in tqdm(objids, desc="Creating RGB composites"):
        i_data, r_data, g_data = make_rgb_components(raw_dir, objid)
        
        if i_data is None:
            continue
            
        # Optional: Crop the center to exclude padding
        # e.g. center crop 224x224 if the original is larger 
        # (Assuming uniform shapes here)
        
        try:
            # make_lupton_rgb is an astronomy standard tool to bring out faint structural details 
            # in galaxies without saturating the bright central cores.
            # This is significantly better than a simple linear stretch!
            rgb_image = make_lupton_rgb(i_data, r_data, g_data, Q=Q, stretch=stretch)
            
            # Save as PNG or JPEG
            out_path = os.path.join(processed_dir, f"{objid}.png")
            Image.fromarray(rgb_image).save(out_path)
            processed_count += 1
        except Exception as e:
            print(f"Error creating RGB for {objid}: {e}")
            
    print(f"Successfully created {processed_count} RGB images in {processed_dir}")

if __name__ == "__main__":
    args = parse_args()
    process_fits_to_rgb(args.raw_dir, args.processed_dir, args.Q, args.stretch)
