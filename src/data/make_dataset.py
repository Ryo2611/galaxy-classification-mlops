import os
import argparse
import pandas as pd
from astropy import coordinates as coords
import astropy.units as u
from astroquery.sdss import SDSS
from tqdm import tqdm
import time
import warnings

# Suppress annoying astropy warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, append=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Download raw SDSS FITS images.")
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to GalaxyZoo CSV with RA/DEC"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/raw", help="Directory to save FITS"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of galaxies to download (0 for all)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="ugriz",
        help="SDSS bands to download (e.g., ugriz)",
    )
    return parser.parse_args()


def download_fits(csv_path: str, output_dir: str, num_samples: int, bands: str):
    """
    Reads coordinates from GalaxyZoo CSV and downloads raw FITS data from SDSS via astroquery.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Optional: Take a random sample or specific subset for quick prototyping
    if num_samples > 0:
        # Priority to confirmed classes rather than uncertain
        df = df[df["UNCERTAIN"] == 0]
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    print(f"Starting download for {len(df)} galaxies...")

    # Create subdirectories for bands
    for b in bands:
        os.makedirs(os.path.join(output_dir, b), exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading FITS"):
        objid = row["OBJID"]
        ra = row["RA"]
        dec = row["DEC"]

        # Parse coordinates (e.g., "00:00:00.41", "-10:22:25.7")
        try:
            pos = coords.SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))

            # Query SDSS for images at this coordinate
            # Using get_images returns a list of HDU objects
            images = SDSS.get_images(coordinates=pos, band=bands)

            if not images:
                print(f"No images found for {objid} at {ra}, {dec}")
                continue

            # Astroquery retrieves all specified bands as a list
            # We match them to the requested bands and save to disk
            for img, band in zip(images, bands):
                save_path = os.path.join(output_dir, band, f"{objid}_{band}.fits")
                # Don't re-download if it already exists
                if not os.path.exists(save_path):
                    img.writeto(save_path, overwrite=True)

            # Avoid hammering the SDSS server too hard
            time.sleep(1.0)

        except Exception as e:
            print(f"Error processing {objid}: {e}")
            continue

    print(f"Done! Raw FITS files are saved in {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    download_fits(args.csv_path, args.output_dir, args.num_samples, args.bands)
