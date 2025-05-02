from numerai_automl.config.config import TARGET_CANDIDATES
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.ensemble_model_manager import EnsembleModelManager


class EnsembleModelPipeline:
    def __init__(self, feature_set, data_manager: DataManager):
        self.feature_set = feature_set
        self.data_manager = data_manager
        self.model_manager = EnsembleModelManager(
            feature_set=feature_set,
            targets_names_for_base_models=TARGET_CANDIDATES
        )
        self.train_data = self.data_manager.load_train_data_for_ensembler()
    
    def run(self):
       self.model_manager.find_weighted_ensemble(
           self.train_data,
           metric="sharpe", 
           number_of_iterations=20, 
           max_number_of_prediction_features_for_ensemble=12, 
           number_of_diffrent_weights_for_ensemble=12)
       print("FINISHED FINDING WEIGHTED ENSEMBLE")
       self.model_manager.find_lgbm_ensemble(
           self.train_data, 
           number_of_iterations=20, 
           cv_folds=5)
       print("FINISHED FINDING LGBM ENSEMBLE")
       