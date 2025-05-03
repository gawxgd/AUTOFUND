from numerai_automl.config.config import TARGET_CANDIDATES
from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.meta_model_manager import MetaModelManager


class MetaModelPipeline:
    def __init__(self, feature_set, data_manager: DataManager):
        self.feature_set = feature_set
        self.data_manager = data_manager
        self.model_manager = MetaModelManager(
            features=self.data_manager.get_features(),
            feature_set=feature_set,
            targets_names_for_base_models=TARGET_CANDIDATES
        )
        self.train_data = self.data_manager.load_train_data_for_ensembler()
    
    def run(self):
      self.model_manager.create_and_save_predictor("weighted")
      print("FINISHED CREATING AND SAVING WEIGHTED PREDICTOR")
      self.model_manager.create_and_save_predictor("lgbm")
      print("FINISHED CREATING AND SAVING LGBM PREDICTOR")
    
      X = self.data_manager.load_validation_data_for_meta_model()
      return X.copy()