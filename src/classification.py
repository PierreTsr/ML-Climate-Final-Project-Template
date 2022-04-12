from shapely.geometry import MultiPolygon

from marida_data_loading import target_mapping, target_categories
from src.outliers_pipeline.plasticfinder.utils import BAND_NAMES, INDICES

NORMED_BANDS = ["NORM_" + band for band in BAND_NAMES]
NORMED_INDICES = ["NORM_" + idx for idx in INDICES]
MEAN_BANDS = ["MEAN_" + band for band in BAND_NAMES]
MEAN_INDICES = ["MEAN_" + idx for idx in INDICES]


def simplify_classes(df, key="id"):
    df.loc[:, key] = df.loc[:, key].map(target_mapping).astype(target_categories)
    return df


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

