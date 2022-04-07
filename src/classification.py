import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon
from multiprocessing import Pool

from tqdm import tqdm

from marida_data_loading import ANOMALIES, CLASSES, marida_categories


def oracle(outlier_df, target_df, category=None):
    if category is not None:
        target_df = target_df.loc[target_df.id == category,]
    target_roi = MultiPolygon(list(target_df.geometry))
    if not target_roi.is_valid:
        target_roi = target_roi.buffer(1e-3)
    oracle_df = outlier_df.loc[outlier_df.geometry.intersects(target_roi),]
    return oracle_df


def align(outlier_df, target_df):
    aligned_df = outlier_df.sjoin(target_df, how="left", predicate="intersects")
    return aligned_df


def buffer_filter(outlier_df, target_df, radius=500):
    target_roi = MultiPolygon(list(target_df.geometry))
    target_roi = target_roi.convex_hull.buffer(radius)
    filtered_df = outlier_df.loc[outlier_df.geometry.intersects(target_roi),]
    return filtered_df


def area_recall(oracle_df, target_df, categories=ANOMALIES, key=None, verbose=False):
    target_df = target_df.loc[target_df.id.isin(categories),]
    if key is not None:
        oracle_df = oracle_df.loc[oracle_df[key] == 1,]

    target_roi = MultiPolygon(list(target_df.geometry))
    if not target_roi.is_valid:
        target_roi = target_roi.buffer(1e-3)
    target_area = target_roi.area
    if target_area == 0:
        return None

    cut_polygons = list(oracle_df.geometry.intersection(target_roi).convex_hull)
    discovered_roi = MultiPolygon(cut_polygons)
    discovered_area = discovered_roi.area

    if verbose:
        print("Target area: {t:.2e} m2, discovered area: {d:.2e} m2, recall: {r:.2e}".format(
            t=target_area,
            d=discovered_area,
            r=discovered_area / target_area
        ))

    return discovered_area / target_area


def pca_plot(df):
    pass
