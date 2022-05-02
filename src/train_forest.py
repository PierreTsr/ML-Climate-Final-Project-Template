import random
import warnings

import geopandas as gpd
from sklearn.model_selection import StratifiedGroupKFold

from marida_data_loading import target_categories
from outliers_pipeline.plasticfinder.utils import BAND_NAMES, INDICES
from src.classification import custom_split, fit_predict_forest
from src.evaluation import compute_classification_kfold

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*zero_division.*")

    print("Loading dataset...")
    full_df = gpd.GeoDataFrame.from_file("data/scenes/marida_outliers.shp")
    normed_bands = ["NORM_" + band for band in BAND_NAMES]
    mean_bands = ["MEAN_" + band for band in BAND_NAMES]
    normed_indices = ["NORM_" + idx for idx in INDICES]
    mean_indices = ["MEAN_" + idx for idx in INDICES]
    full_df.id = full_df.id.astype(target_categories)
    print("Done!")

    cv = StratifiedGroupKFold(n_splits=5)
    train_df, val_df = custom_split(full_df)

    random_param = [{
        "n_estimators": random.choice([10, 50, 100, 200, 500, 1000]),
        "n_jobs": 15,
        "criterion": random.choice(["gini", "entropy"]),
        "max_depth": random.choice([2, 5, 10, 20, 50]),
        "min_samples_split": random.choice([2, 5, 10, 50, 100, 200, 500]),
        "min_samples_leaf": random.choice([2, 5, 10, 50, 100, 200, 500]),
        "max_samples": random.choice([0.1, 0.2, 0.5, 0.7, 0.9]),
        "max_features": random.choice([0.1, 0.2, 0.5, 0.7, 0.9])
    } for _ in range(500)]

    print("Training...")
    df = compute_classification_kfold(train_df,
                                      cv,
                                      models={"Forest": fit_predict_forest},
                                      kwargs=random_param)
    print("Done!")
    df.to_csv("etc/forest_training_results.csv")
