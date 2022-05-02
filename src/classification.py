from pathlib import Path
import random

import pandas as pd
from shapely.geometry import MultiPolygon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from marida_data_loading import target_mapping, target_categories
from src.outliers_pipeline.plasticfinder.utils import BAND_NAMES, INDICES, get_matching_marida_target

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


def load_marida_split(marida_dir=Path("data/MARIDA")):
    train_file = marida_dir / "splits" / "train_X.txt"
    test_file = marida_dir / "splits" / "test_X.txt"
    val_file = marida_dir / "splits" / "val_X.txt"
    results = []
    for filename in (train_file, test_file, val_file):
        with open(filename) as file:
            scenes = set()
            for line in file.readlines():
                scene = "S2_" + "_".join(line.split("_")[:-1])
                scenes.add(scene)
            results.append(scenes)
    return results


def load_custom_split(marida_dir=Path("data/MARIDA")):
    train_file = marida_dir / "custom_split" / "train_X.txt"
    val_file = marida_dir / "custom_split" / "val_X.txt"
    results = []
    for filename in (train_file, val_file):
        with open(filename) as file:
            scenes = set()
            for line in file.readlines():
                scene = "_".join(line[:-1].split("_"))
                scenes.add(scene)
            results.append(scenes)
    return results


def random_dataset_split(full_df, seed=0, split=(0.8, 0.2), tol=3e-2):
    random.seed(seed)
    counts = full_df.loc[full_df.id == "Non-Organic Debris", ["scene", "id"]].groupby("scene").count()
    total = full_df.loc[full_df.id == "Non-Organic Debris"].shape[0]
    tiles = list(set(full_df.scene.values))
    tiles.sort()
    train, val = [], []
    train_ct, val_ct = 0, 0
    for tile in tiles:
        b = bool(random.getrandbits(1))
        try:
            c = counts.loc[tile, "id"]
        except KeyError:
            c = 0
        if b and val_ct + c < total * (split[1] + tol):
            val.append(tile)
            val_ct += c
        else:
            train.append(tile)
            train_ct += c
    train_df = full_df.loc[full_df.scene.isin(train), :]
    val_df = full_df.loc[full_df.scene.isin(val), :]
    return train_df, val_df


def marida_split(full_df, marida_dir=Path("data/MARIDA")):
    train_scenes, test_scenes, val_scenes = load_marida_split(marida_dir)
    scenes = full_df.patch.apply(lambda p: get_matching_marida_target(Path(p).parent.parent.stem))
    train_df = full_df.loc[scenes.isin(train_scenes),:]
    test_df = full_df.loc[scenes.isin(test_scenes),:]
    val_df = full_df.loc[scenes.isin(val_scenes),:]
    return train_df, test_df, val_df


def custom_split(full_df, marida_dir=Path("data/MARIDA")):
    train_scenes, val_scenes = load_custom_split(marida_dir)
    scenes = full_df.patch.apply(lambda p: get_matching_marida_target(Path(p).parent.parent.stem))
    train_df = full_df.loc[scenes.isin(train_scenes),:]
    val_df = full_df.loc[scenes.isin(val_scenes),:]
    return train_df, val_df


def fit_predict_lda(train_df, test_df, features, **kwargs):
    lda = LinearDiscriminantAnalysis(**kwargs)
    lda.fit(train_df.loc[:, features].values, train_df.id.values)
    pred = lda.predict(test_df.loc[:, features].values)
    return pd.DataFrame(classification_report(test_df.id.values, pred, output_dict=True)).transpose()


def fit_predict_gnb(train_df, test_df, features, **kwargs):
    gnb = GaussianNB(**kwargs)
    gnb.fit(train_df.loc[:, features].values, train_df.id.values)
    pred = gnb.predict(test_df.loc[:, features].values)
    return pd.DataFrame(classification_report(test_df.id.values, pred, output_dict=True)).transpose()


def fit_predict_forest(train_df, test_df, features, **kwargs):
    forest = RandomForestClassifier(**kwargs)
    forest.fit(train_df.loc[:, features].values, train_df.id.values)
    pred = forest.predict(test_df.loc[:, features].values)
    return pd.DataFrame(classification_report(test_df.id.values, pred, output_dict=True)).transpose()


def fit_predict_adaboost(train_df, test_df, features, **kwargs):
    adaboost = AdaBoostClassifier(**kwargs)
    adaboost.fit(train_df.loc[:, features].values, train_df.id.values)
    pred = adaboost.predict(test_df.loc[:, features].values)
    return pd.DataFrame(classification_report(test_df.id.values, pred, output_dict=True)).transpose()
