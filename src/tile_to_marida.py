from argparse import ArgumentParser
from itertools import compress
from pathlib import Path

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
    args = parser.parse_args()

    tiles_dir = Path(args.t)
    tiles = list(tiles_dir.glob("*.tif"))
    tiles_scenes = [get_matching_marida_target(tile.stem) for tile in tiles]

    marida_dir = Path(args.m)
    marida_scenes = set(scene.stem for scene in (marida_dir / "shapefiles").glob("*.shp"))

    matching = [scene in marida_scenes for scene in tiles_scenes]
    tiles = list(compress(tiles, matching))

    print("Found {} matching tiles: {}".format(len(tiles), [tile.stem for tile in tiles]))

    output_dir = Path(args.o)

    for tile in tiles:
        scene = marida_dir / "shapefiles" / (get_matching_marida_target(tile.stem) + ".shp")
        roi = marida_dir / "ROI" / (scene.stem + ".geojson")

        pre_process_tile(output_dir, tile.stem, tiles_dir, patches=(20, 20), roi=roi)
        pre_processing_visualizations(output_dir / tile.stem)
        outliers_df = post_process_patches(output_dir / tile.stem)
        post_processing_visualizations(output_dir / tile.stem)

        marida_df = load_target(scene)
        aligned_df = align(outliers_df, marida_df)
        aligned_df.to_file(output_dir / tile.stem / "aligned_outliers.shp")
