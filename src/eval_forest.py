import numpy as np
import pandas as pd
import geopandas as gpd

from src.classification import MEAN_INDICES, NORMED_INDICES, MEAN_BANDS, NORMED_BANDS, custom_split, fit_predict_forest
from src.marida_data_loading import target_categories
from src.outliers_pipeline.plasticfinder.utils import BAND_NAMES, INDICES

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

params = ["n_estimators", "n_jobs", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "max_samples",
          "max_features"]


shared = {
    "n_estimators": 200,
    "n_jobs": 15,
    "criterion": "entropy",
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "max_samples": 0.9,
    "max_features": 0.5,
}

def eval_row(row, train_df, val_df):
    f = features[row["features"]]
    kwargs = row[params]
    res = fit_predict_forest(train_df, val_df, f, **kwargs)
    return res


if __name__ == "__main__":
    models = pd.read_csv("etc/forest_training_results.csv", index_col=list(range(len(params) + 3)))
    l = list(models.index.names)
    l.remove("fold")
    models_mean = models.groupby(l).mean().reset_index()
    models_std = models.groupby(l).std().reset_index()

    full_fd = gpd.GeoDataFrame.from_file("data/scenes/marida_outliers.shp")
    train_df, val_df = custom_split(full_fd)

    l.remove("category")
    l.remove("features")

    eval_df = pd.DataFrame(np.nan,
                           index=pd.MultiIndex.from_product((
                               list(set(models_mean["features"].values)),
                               [*target_categories.categories, "accuracy", "macro avg"]
                           ), names=["features", "category"]),
                           columns=pd.MultiIndex.from_product((
                               ["Random Search", "Shared"],
                               ["precision", "recall", "f1-score"]))
                           )

    best_models_plastic = models_mean.loc[models_mean.category == "Non-Organic Debris", :]
    idx = best_models_plastic.groupby("features", sort=False)["f1-score"].transform(max) == best_models_plastic[
        "f1-score"]
    best_models_plastic = best_models_plastic.loc[idx, :]
    for idx, row in best_models_plastic.iterrows():
        eval = eval_row(row, train_df, val_df)
        eval.index = pd.MultiIndex.from_product(((row["features"],), eval.index))
        eval.columns = pd.MultiIndex.from_product((("Random Search", ), eval.columns))
        eval_df.update(eval)

    for name in list(set(models_mean["features"].values)):
        f = features[name]
        eval = fit_predict_forest(train_df, val_df, f, **shared)
        eval.index = pd.MultiIndex.from_product(((name,), eval.index))
        eval.columns = pd.MultiIndex.from_product((("Shared", ), eval.columns))
        eval_df.update(eval)

    eval_df.to_csv("etc/forest_evaluation.csv")
