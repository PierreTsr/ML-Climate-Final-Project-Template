from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from fiona.errors import DriverError
from matplotlib import pyplot as plt
from shapely.geometry import MultiPolygon
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.classification import NORMED_INDICES, MEAN_INDICES, NORMED_BANDS, MEAN_BANDS, fit_predict_lda, \
    fit_predict_forest, fit_predict_adaboost, fit_predict_gnb
from src.marida_data_loading import target_mapping, CLASSES, load_target, marida_categories, target_categories, \
    ANOMALIES, COLOR_MAPPING
from src.outliers_pipeline.plasticfinder.utils import get_matching_marida_target, BAND_NAMES, INDICES, compute_tile_size


def total_recall(tile_dir=Path("data/scenes"), marida_dir=Path("data/MARIDA"), outliers=("LOCAL", "GLOBAL", "FOREST"),
                 simple_targets=False):
    tiles = list(set(path.stem for path in tile_dir.iterdir()))
    if simple_targets:
        categories = target_mapping.values()
    else:
        categories = CLASSES
    areas = pd.DataFrame(0,
                         index=pd.MultiIndex.from_product((tiles, categories), names=("tile", "target")),
                         columns=[*outliers, "TARGET"])
    for tile in tiles:
        scene = get_matching_marida_target(tile)
        print(scene)
        try:
            outlier_df = gpd.GeoDataFrame.from_file(tile_dir / tile / "aligned_outliers.shp").dropna()
            target_df = load_target(marida_dir / "shapefiles" / (scene + ".shp"))
        except DriverError:
            continue
        outlier_df.id = outlier_df.id.astype(marida_categories)
        if simple_targets:
            outlier_df.id = outlier_df.id.map(target_mapping).astype(target_categories)
            target_df.id = target_df.id.map(target_mapping).astype(target_categories)
        for cat in categories:
            for key in outliers:
                discovered, target = area_recall(outlier_df, target_df, categories=[cat], key=key)
                areas.loc[(tile, cat), key] = discovered
                areas.loc[(tile, cat), "TARGET"] = target
    return areas


def area_recall(oracle_df, target_df, categories=ANOMALIES, key=None, verbose=False):
    target_df = target_df.loc[target_df.id.isin(categories),]
    if key is not None:
        oracle_df = oracle_df.loc[oracle_df[key] == 1,]

    target_roi = MultiPolygon(list(target_df.geometry))
    if not target_roi.is_valid:
        target_roi = target_roi.buffer(1e-3)
    target_area = target_roi.area
    if target_area == 0:
        return 0, 0

    cut_polygons = list(oracle_df.geometry.intersection(target_roi))
    tmp = []
    for poly in cut_polygons:
        if isinstance(poly, MultiPolygon):
            tmp += poly.geoms
        else:
            tmp.append(poly)
    cut_polygons = tmp
    discovered_roi = MultiPolygon(cut_polygons)
    discovered_area = discovered_roi.area

    if verbose:
        print("Target area: {t:.2e} m2, discovered area: {d:.2e} m2, recall: {r:.2e}".format(
            t=target_area,
            d=discovered_area,
            r=discovered_area / target_area
        ))

    return discovered_area, target_area


def plot_scatter(df, key_1, key_2, outliers="LOCAL", use_unidentified=False, filename=None, classes=CLASSES,
                 colors=COLOR_MAPPING):
    df = df.loc[df[outliers] == 1, :]

    fig, ax = plt.subplots(figsize=(15, 10))
    s = 12
    if use_unidentified:
        na = df.loc[df.id.isna(), :]
        ax.scatter(na[key_1], na[key_2], c="#000000", label="NA", s=s)
    for cat, col in colors.items():
        if cat not in classes:
            continue
        tmp = df.loc[df.id == cat, :]
        ax.scatter(tmp.loc[:, key_1], tmp.loc[:, key_2], c=col, label=cat, s=s)
    ax.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.plot()
    return fig, ax


def compare_explained_variance(train_df, test_df, filename=None, target_names=None):
    pca = PCA()
    lda = LinearDiscriminantAnalysis()

    target = train_df.id.cat.codes
    features1 = train_df.loc[:, BAND_NAMES + INDICES]
    features2 = train_df.loc[:, BAND_NAMES + NORMED_INDICES + MEAN_INDICES]
    features3 = train_df.loc[:, NORMED_BANDS + INDICES + MEAN_BANDS]
    features4 = train_df.loc[:, NORMED_BANDS + NORMED_INDICES + MEAN_BANDS + MEAN_INDICES]

    fig, axs = plt.subplots(figsize=(10, 6))

    n = 15
    pca.fit(features1.values)
    axs.plot(range(1, n + 1), np.cumsum(pca.explained_variance_ratio_)[:n], c="blue", linestyle="-",
             label="no normalization")
    pca.fit(features2.values)
    axs.plot(range(1, n + 1), np.cumsum(pca.explained_variance_ratio_)[:n], c="red", linestyle="--",
             label="normalized indices")
    pca.fit(features3.values)
    axs.plot(range(1, n + 1), np.cumsum(pca.explained_variance_ratio_)[:n], c="green", linestyle="-.",
             label="normalized bands")
    pca.fit(features4.values)
    axs.plot(range(1, n + 1), np.cumsum(pca.explained_variance_ratio_)[:n], c="black", linestyle=":",
             label="normalized bands and indices")

    axs.legend()

    lda.fit(features1.values, target.values)
    pred = lda.predict(test_df.loc[:, BAND_NAMES + INDICES].values)
    print(classification_report(test_df.id.cat.codes.values, pred, target_names=target_names))
    lda.fit(features2.values, target.values)
    pred = lda.predict(test_df.loc[:, BAND_NAMES + NORMED_INDICES + MEAN_INDICES].values)
    print(classification_report(test_df.id.cat.codes.values, pred, target_names=target_names))
    lda.fit(features3.values, target.values)
    pred = lda.predict(test_df.loc[:, NORMED_BANDS + INDICES + MEAN_BANDS].values)
    print(classification_report(test_df.id.cat.codes.values, pred, target_names=target_names))
    lda.fit(features4.values, target.values)
    pred = lda.predict(test_df.loc[:, NORMED_BANDS + NORMED_INDICES + MEAN_BANDS + MEAN_INDICES].values)
    print(classification_report(test_df.id.cat.codes.values, pred, target_names=target_names))

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.plot()
    return fig, axs


def compute_reduction(scene_dir=Path("data/scenes"), outliers=("LOCAL", "GLOBAL", "FOREST")):
    scenes = [scene.stem for scene in scene_dir.iterdir()]
    reduction_df = pd.DataFrame(0, index=scenes, columns=["ORIGINAL", *outliers])
    for scene in scene_dir.iterdir():
        print(scene.stem)
        try:
            aligned_df = gpd.GeoDataFrame.from_file(scene / "aligned_outliers.shp")
            original_size = compute_tile_size(scene)
            reduction_df.loc[scene.stem, "ORIGINAL"] = original_size
            reduction_df.loc[scene.stem, outliers] = aligned_df.loc[:, outliers].sum(axis=0)
        except (DriverError, KeyError) as e:
            print(e)
            continue
    return reduction_df


def compute_validation(train_df, test_df,
                       models={"GNB": fit_predict_gnb,
                               "LDA": fit_predict_lda},
                       features=None
                       ):
    if features is None:
        features = {
            "bands": BAND_NAMES,
            "bands_with_means": BAND_NAMES + MEAN_BANDS,
            "indices": INDICES,
            "indices_with_means": INDICES + MEAN_INDICES,
            "standard": BAND_NAMES + INDICES,
            "standard_with_means": BAND_NAMES + INDICES + MEAN_BANDS + MEAN_INDICES,
            "normalized_indices": BAND_NAMES + NORMED_INDICES,
            "normalized_indices_with_means": BAND_NAMES + NORMED_INDICES + MEAN_INDICES,
            "normalized_bands": NORMED_BANDS + INDICES,
            "normalized_bands_with_means": NORMED_BANDS + INDICES + MEAN_BANDS,
            "normalized": NORMED_BANDS + NORMED_INDICES,
            "normalized_with_means": NORMED_BANDS + MEAN_BANDS + NORMED_INDICES + MEAN_INDICES,
            "all": NORMED_BANDS + NORMED_INDICES + BAND_NAMES + INDICES + MEAN_BANDS + MEAN_INDICES
        }

    df = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product((models.keys(),
                                          [*target_categories.categories, "accuracy", "macro avg"]),
                                         names=["model", "category"]),
        columns=pd.MultiIndex.from_product((features.keys(), ["precision", "recall", "f1-score"]))
    )

    for name, feature in features.items():
        for model, func in models.items():
            res = func(train_df, test_df, feature)
            res.index = pd.MultiIndex.from_product(((model,), res.index))
            res.columns = pd.MultiIndex.from_product(((name,), res.columns))
            df.update(res)
    return df


def compute_classification_kfold(full_df, cv,
                                 models={"GNB": fit_predict_gnb,
                                         "LDA": fit_predict_lda},
                                 features=None,
                                 kwargs=None
                                 ):
    if kwargs is not None and len(models) > 1:
        raise NotImplementedError("cannot have multiple models with specified arguments.")

    if features is None:
        features = {
            "bands": BAND_NAMES,
            # "bands_with_means": BAND_NAMES + MEAN_BANDS,
            "indices": INDICES,
            # "indices_with_means": INDICES + MEAN_INDICES,
            "standard": BAND_NAMES + INDICES,
            # "standard_with_means": BAND_NAMES + INDICES + MEAN_BANDS + MEAN_INDICES,
            "normalized_indices": BAND_NAMES + NORMED_INDICES,
            # "normalized_indices_with_means": BAND_NAMES + NORMED_INDICES + MEAN_INDICES,
            "normalized_bands": NORMED_BANDS + INDICES,
            # "normalized_bands_with_means": NORMED_BANDS + INDICES + MEAN_BANDS,
            "normalized": NORMED_BANDS + NORMED_INDICES,
            # "normalized_with_means": NORMED_BANDS + MEAN_BANDS + NORMED_INDICES + MEAN_INDICES,
            "all": NORMED_BANDS + NORMED_INDICES + BAND_NAMES + INDICES + MEAN_BANDS + MEAN_INDICES
        }

    if kwargs is None:
        df = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_product((models.keys(),
                                              features.keys(),
                                              [i for i in range(cv.n_splits)],
                                              ["Non-Organic Debris", "accuracy", "macro avg"]),
                                             names=["model", "features", "fold", "category"]),
            columns=["precision", "recall", "f1-score"]
        )
    else:
        standard_index = pd.MultiIndex.from_product((features.keys(),
                                                     range(cv.n_splits),
                                                     ["Non-Organic Debris", "accuracy", "macro avg"]),
                                                    names=["features", "fold", "category"])
        args_index = pd.MultiIndex.from_arrays((
            [kwargs[i][key] for i in range(len(kwargs))] for key in kwargs[0].keys()),
            names=kwargs[0].keys()
        )
        df = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(((*arg, *idx) for arg in args_index for idx in standard_index),
                                            names=args_index.names + standard_index.names),
            columns=["precision", "recall", "f1-score"]
        )

    for i, (train_idx, test_idx) in tqdm(enumerate(cv.split(full_df, full_df.id, full_df.scene)), desc="fold", position=0):
        train_df = full_df.iloc[train_idx, :]
        test_df = full_df.iloc[test_idx, :]
        for name, feature in tqdm(features.items(), desc="features", position=1, leave=False):
            if len(models) > 1:
                for model, func in models.items():
                    res = func(train_df, test_df, feature)
                    res.index = pd.MultiIndex.from_product(((model,), (name,), (i,), res.index))
                    df.update(res)
            elif kwargs is not None:
                model, func = list(models.items())[0]
                for kwarg in tqdm(kwargs, desc="param", position=2, leave=False):
                    res = func(train_df, test_df, feature, kwarg)
                    res.index = pd.MultiIndex.from_tuples(((*kwarg.values(), name, i, key) for key in res.index))
                    df.update(res)
    return df
