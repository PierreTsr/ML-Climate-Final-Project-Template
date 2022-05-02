from argparse import ArgumentParser
from itertools import compress
from pathlib import Path
import geopandas as gpd
import pandas as pd
from fiona.errors import DriverError

from classification import align
from marida_data_loading import load_target
from outliers_pipeline.plasticfinder.data_processing import pre_processing_visualizations, post_process_patches, \
    post_processing_visualizations
from outliers_pipeline.plasticfinder.data_querying import pre_process_tile
from outliers_pipeline.plasticfinder.utils import get_matching_marida_target


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This script finds all the tiles in .tiff format in a directory, that are matching a MARIDA scene. It then computes their outliers, the matching pixels with MARIDA, and wirtes both datasets.")
    parser.add_argument("-t", type=str, help="Tiles directory")
    parser.add_argument("-m", type=str, help="MARIDA directory")
    parser.add_argument("-o", type=str, help="output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tiles_dir = Path(args.t)
    tiles = list(tiles_dir.glob("*.tif"))
    tiles_scenes = [get_matching_marida_target(tile.stem) for tile in tiles]

    marida_dir = Path(args.m)
    marida_scenes = set(scene.stem for scene in (marida_dir / "shapefiles").glob("*.shp"))

    matching = [scene in marida_scenes for scene in tiles_scenes]
    tiles = list(compress(tiles, matching))

    output_dir = Path(args.o)

    if not args.overwrite:
        print("Looking for already processed scenes...")
        existing_tiles = set([path.stem for path in output_dir.iterdir()])
        remaining_tiles = []
        for tile in tiles:
            if tile.stem in existing_tiles:
                continue
            remaining_tiles.append(tile)
        tiles = remaining_tiles

    print("Found {} matching tiles: {}".format(len(tiles), [tile.stem for tile in tiles]))

    marida_outliers = []
    try:
        gdf = gpd.GeoDataFrame.from_file(output_dir / "marida_outliers.shp")
        print("Found exisiting outlier df, adding new data to it.")
        marida_outliers.append(gdf)
    except DriverError:
        pass

    for tile in tiles:
        scene = marida_dir / "shapefiles" / (get_matching_marida_target(tile.stem) + ".shp")
        roi = marida_dir / "ROI" / (scene.stem + ".geojson")

        pre_process_tile(output_dir, tile.stem, tiles_dir, patches=(20, 20), roi=roi)
        pre_processing_visualizations(output_dir / tile.stem)
        outliers_df = post_process_patches(output_dir / tile.stem, outliers_keys=("GLOBAL_OUTLIERS", ))
        post_processing_visualizations(output_dir / tile.stem)

        if outliers_df is None:
            continue

        marida_df = load_target(scene)
        aligned_df = align(outliers_df, marida_df)
        aligned_df.loc[:, "id"] = aligned_df.id.astype(str)
        aligned_df.to_file(output_dir / tile.stem / "aligned_outliers.shp")

        marida_outliers.append(aligned_df.dropna())

    marida_outliers = pd.concat(marida_outliers)
    marida_outliers.to_file(output_dir / "marida_outliers.shp")
