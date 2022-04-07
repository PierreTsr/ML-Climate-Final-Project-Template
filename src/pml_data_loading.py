import pandas as pd
import geopandas as gpd
import re
from pathlib import Path
from shapely.geometry import MultiPoint

from outliers_pipeline.plasticfinder.utils import get_tile_bounding_box


def observations_roi(features_file, radius=1000, target_dir=None):
    features_path = Path(features_file)
    tile_name = re.search("[0-9]{2}[A-Z]{3}", features_path.name)[0]
    _, utm = get_tile_bounding_box(tile_name)

    df = pd.read_csv(features_path, sep="\t")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs=4326).to_crs(utm)
    hull = MultiPoint(gdf.geometry).convex_hull()
    hull = hull.buffer(radius)

    if target_dir is not None:
        target_file = Path(target_dir) / (features_path.stem + ".geojson")
        s = gpd.GeoSeries([hull]).set_crs(utm)
        s.to_file(str(target_file), driver="GeoJSON")

    return hull, utm


def find_matching_pixels(tile_df, target_df, col="category"):
    hull, utm = observations_roi(target_df, 20)

    df = pd.read_csv(target_df, sep="\t")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon, df.Lat), crs=4326).to_crs(utm)

    tile_df["category"] = pd.NA
    neighbors = tile_df.loc[tile_df.geometry.within(hull), ]

    count = 0
    for target in gdf:
        matching = neighbors.loc[neighbors.geometry.contains(target.geometry)]
        if matching.shape[0] == 0:
            continue
        count += 1
        matching["category"] = gdf[col]

    print("Found {} targets out of {}.".format(count, gdf.shape[0]))
    return tile_df
