import os
import pandas as pd
from numerai_automl.config.config import TARGET_CANDIDATES
from numerai_automl.data_managers.data_downloader import DataDownloader
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.pipeline.ensemble_model_pipeline import EnsembleModelPipeline
from numerai_automl.pipeline.meta_model_pipeline import MetaModelPipeline
from numerai_automl.raport_manager.raport_manager import RaportManager
from numerai_automl.scorer.scorer import Scorer
from numerai_automl.utils.utils import get_project_root
from base_model_pipeline import BaseModelPipeline
from numerai_automl.visual.cumsum_cor_plot import CumSumCorPlot
from numerai_automl.visual.radar_plot import RadarPlot


class MainPipeline:
    def __init__(self, data_version="v5.0", feature_set="medium"):
        self.data_version = data_version
        self.feature_set = feature_set
        self.data_manager = DataManager(data_version, feature_set)

    def run(self):
        self.download_data()
        base_model_pipeline = BaseModelPipeline(self.feature_set, self.data_manager)
        base_model_pipeline.run()

        ensemble_model_pipeline = EnsembleModelPipeline(self.feature_set, self.data_manager)
        ensemble_model_pipeline.run()

        meta_model_pipeline = MetaModelPipeline(self.feature_set, self.data_manager)
        X = meta_model_pipeline.run()
        return_data = X.copy()

        print("FINISHED LOADING DATA FOR META MODEL")
        # now predicting
        predictor_weighted = meta_model_pipeline.model_manager.load_predictor("weighted")
        predictor_lgbm = meta_model_pipeline.model_manager.load_predictor("lgbm")
        return_data["predictions_model_meta_weighted"] = predictor_weighted(X)
        return_data["predictions_model_meta_lgbm"] = predictor_lgbm(X)
        return_data["predictions_model_omega"] = (return_data[["predictions_model_meta_weighted", "predictions_model_meta_lgbm"]].sum(axis=1)) / 2

        base_models_predictors = BaseModelPipeline.model_manager.load_base_model_predictors()
        neutralized_base_models_predictors = BaseModelPipeline.load_neutralized_base_model_predictors()

        for target_name in TARGET_CANDIDATES:
            return_data[f"predictions_model_{target_name}"] = base_models_predictors[f"model_{target_name}"](X)
            return_data[f"neutralized_predictions_model_{target_name}"] = neutralized_base_models_predictors[f"neutralized_model_{target_name}"](X)
        
        print("FINISHED PREDICTING")

        return_data.to_csv("return_data.csv")

        scorer = Scorer()
        scores = scorer.compute_scores(return_data, "target")

        scores.to_csv("return_data_for_scoring.csv")

        print(f"FINISHED SCORE COMPUTING")

        df = return_data[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega", 'era',  "target"]]
        csp = CumSumCorPlot(df)
        cumsum_cor_plot = csp.get_plot()
        df = pd.read_csv("return_data_for_scoring.csv", index_col=0)
        df = df.loc[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", "predictions_model_omega"]]
        rp = RadarPlot(df)
        radar_plot = rp.get_plot()
        rm = RaportManager([cumsum_cor_plot, radar_plot])
        rm.generate_html("raport.html")

        print("FINISHED GENERATING RAPORT")

def download_data(data_version: str):
    """
    Check if required data files exist; if not, download them using DataDownloader.

    Args:
        data_version (str): The version of the dataset to work with.
    """
    required_files = ["features.json", "train.parquet", "validation.parquet", "live.parquet"]
    project_root = get_project_root()
    data_path = os.path.join(project_root, data_version)

    missing_files = [
        f for f in required_files
        if not os.path.exists(os.path.join(data_path, f))
    ]

    if not missing_files:
        print(f"[INFO] All required data files for version '{data_version}' already exist.")
        return

    print(f"[INFO] Missing files detected for version '{data_version}': {missing_files}")
    print("[INFO] Initiating download of missing data files...")
    downloader = DataDownloader(data_version)
    downloader.download_all_data()
    print("[INFO] Download complete.")   
