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

COLOR_MAPPING = {
    "Marine Debris": "#ff2a00",
    "Dense Sargassum": "#013602",
    "Sparse Sargassum": "#00e007",
    "Natural Organic Material": "#65801c",
    "Marine Water": "#001ba1",
    "Sediment-Laden Water": "#204d4b",
    "Foam": "#639cf7",
    "Turbid Water": "#08474d",
    "Shallow Water": "#366387",
    "Waves": "#386aff",
    "Mixed Water": "#667a8a",
    "Wakes": "#5a81a1",
    "Ship": "#ff00ea",
    "Cloud Shadows": "#2e2e2d",
    "Clouds": "#a6a6a6",
}


PLASTIC = ["Marine Debris"]

ANOMALIES = [
    "Marine Debris",
    "Dense Sargassum",
    "Natural Organic Material",
    "Ship",
    "Sediment-Laden Water",
    "Cloud Shadows",
]

target_mapping = {
    "Marine Debris": "Non-Organic Debris",
    "Dense Sargassum": "Sargassum",
    "Sparse Sargassum": "Sargassum",
    "Natural Organic Material": "Organic Debris",
    "Ship": "Ship",
    "Clouds": "Clouds",
    "Marine Water": "Other Water types",
    "Sediment-Laden Water": "Other Water types",
    "Foam": "Other Water types",
    "Turbid Water": "Other Water types",
    "Shallow Water": "Other Water types",
    "Waves": "Other Water types",
    "Cloud Shadows": "Cloud Shadows",
    "Wakes": "Other Water types",
    "Mixed Water": "Other Water types"
}

marida_categories = pd.CategoricalDtype(categories=CLASSES, ordered=False)
target_categories = pd.CategoricalDtype(categories=set(target_mapping.values()), ordered=False)
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
