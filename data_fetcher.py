"""
Satellite Imagery Data Fetcher

This script downloads satellite images for properties using their latitude and longitude coordinates.
It uses the ESRI World Imagery tile service to fetch high-resolution satellite imagery.

Usage:
    python data_fetcher.py --csv_path "data/train(1)(train(1)).csv" --output_dir "property_images_v2"
"""

import os
import math
import requests
import pandas as pd
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def deg2num(lat_deg, lon_deg, zoom):
    """
    Converts latitude/longitude coordinates to XYZ tile coordinates.
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        zoom: Zoom level (higher = more detail, typically 15-19)
    
    Returns:
        Tuple of (xtile, ytile) coordinates
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def download_tile_direct(row, save_dir, zoom=19):
    """
    Downloads a single satellite image tile for a property.
    
    Args:
        row: DataFrame row containing 'id', 'lat', and 'long' columns
        save_dir: Directory to save images
        zoom: Zoom level for the satellite imagery
    
    Returns:
        True if successful, False otherwise
    """
    prop_id = row['id']
    save_path = os.path.join(save_dir, f"{prop_id}.png")
    
    # Skip if image already exists
    if os.path.exists(save_path):
        return True
    
    try:
        # Convert lat/lon to tile coordinates
        xtile, ytile = deg2num(row['lat'], row['long'], zoom)
        
        # ESRI World Imagery Tile URL
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        
        # Download the tile
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading image for property {prop_id}: {e}")
        return False


def download_images(csv_path, output_dir, zoom=19, threads=20):
    """
    Main function to download satellite images for all properties in the CSV.
    
    Args:
        csv_path: Path to CSV file containing property data with 'id', 'lat', 'long' columns
        output_dir: Directory to save downloaded images
        zoom: Zoom level for satellite imagery (default: 19 for high detail)
        threads: Number of parallel download threads (default: 20)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} properties")
    
    # Verify required columns exist
    required_cols = ['id', 'lat', 'long']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check how many images already exist
    existing_count = sum(1 for prop_id in df['id'] 
                        if os.path.exists(os.path.join(output_dir, f"{prop_id}.png")))
    print(f"Images already downloaded: {existing_count}/{len(df)}")
    
    # Download images
    print(f"Starting download (Zoom level: {zoom}, Threads: {threads})...")
    print(f"Output directory: {output_dir}")
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(tqdm(
            executor.map(
                lambda row: download_tile_direct(row, output_dir, zoom),
                df.to_dict('records')
            ),
            total=len(df),
            desc="Downloading images"
        ))
    
    # Summary
    successful = sum(results)
    failed = len(results) - successful
    print(f"\nDownload complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Images saved to: {output_dir}")


def main():
    """Command-line interface for the data fetcher."""
    parser = argparse.ArgumentParser(
        description="Download satellite imagery for properties using lat/lon coordinates"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train(1)(train(1)).csv",
        help="Path to CSV file with property data (default: data/train(1)(train(1)).csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="property_images_v2",
        help="Directory to save downloaded images (default: property_images_v2)"
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Zoom level for satellite imagery (15-19, default: 19)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="Number of parallel download threads (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Validate zoom level
    if args.zoom < 10 or args.zoom > 20:
        print("Warning: Zoom level should typically be between 15-19 for best results")
    
    # Run the download
    download_images(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        zoom=args.zoom,
        threads=args.threads
    )


if __name__ == "__main__":
    main()

