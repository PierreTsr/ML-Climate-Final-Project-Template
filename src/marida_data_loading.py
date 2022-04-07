import pandas as pd
import geopandas as gpd
import re
from datetime import date
from argparse import ArgumentParser
from pathlib import Path
from shapely.geometry import MultiPolygon

CLASSES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water"
]


PLASTIC = ["Marine Debris"]

ANOMALIES = [
    "Marine Debris",
    "Dense Sargassum",
    "Natural Organic Material",
    "Ship",
    "Sediment-Laden Water",
    "Cloud Shadows",
]

marida_categories = pd.CategoricalDtype(categories=CLASSES, ordered=True)
marida_mapping = {i + 1: cat for i, cat in enumerate(CLASSES)}


def get_roi(path, radius, target_dir=None):
    gdf = gpd.GeoDataFrame.from_file(str(path))
    hull = MultiPolygon(list(gdf.geometry)).convex_hull.buffer(radius)

    if target_dir is not None:
        target_file = target_dir / (path.stem + ".geojson")
        s = gpd.GeoSeries([hull])
        s.set_crs(gdf.crs)
        s.to_file(str(target_file), driver="GeoJSON")
    return hull


def set_all_roi(base_dir, target_dir, radius=1000):
    target_dir.mkdir(exist_ok=True)
    for path in base_dir.glob("*.shp"):
        get_roi(path, radius, target_dir)


def load_target(path):
    gdf = gpd.GeoDataFrame.from_file(path)
    gdf.id = gdf.id.map(marida_mapping).astype(marida_categories)
    return gdf


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute geojson ROIs for the MARIDA dataset targets")
    parser.add_argument("--marida", type=str, help="directory of the MARIDA shapefile targets")
    parser.add_argument("--dst", type=str, help="directory where the ROIs will be saved")
    args = parser.parse_args()

    base_dir = Path(args.marida)
    target_dir = Path(args.dst)

    set_all_roi(base_dir, target_dir)
