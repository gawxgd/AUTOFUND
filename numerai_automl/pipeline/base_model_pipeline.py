from numerai_automl.data_managers.data_manager import DataManager
from numerai_automl.model_managers.base_model_manager import BaseModelManager
from numerai_automl.config.config import TARGET_CANDIDATES


class BaseModelPipeline:
    def __init__(self, feature_set, data_manager: DataManager):
        self.feature_set = feature_set
        self.data_manager = data_manager
        self.model_manager = BaseModelManager(
            feature_set=feature_set,
            targets_names_for_base_models=TARGET_CANDIDATES
        )
        self.train_data = self.data_manager.load_train_data_for_base_models()
    
    def run(self):
        self.train_and_save(self.train_data)
        self.create_and_save_predictions()
        self.neutralize_predictions()

    def train_and_save(self, train_data):
        self.model_manager.train_base_models(train_data)
        print("FINISHED TRAINING BASE MODELS")
        self.model_manager.save_base_models()
        print("FINISHED SAVING BASE MODELS")

    def create_and_save_predictions(self):
        prediction_data = self.data_manager.load_data_for_creating_predictions_for_base_models()
        predictions = self.model_manager.create_predictions_by_base_models(prediction_data)
        self.data_manager.save_vanila_predictions_by_base_models(predictions)
        print("FINISHED CREATING PREDICTIONS BY BASE MODELS")

    def neutralize_predictions(self):
        validation_data = self.data_manager.load_vanila_predictions_data_by_base_models()
        self.model_manager.find_neutralization_features_and_proportions_for_base_models(
            validation_data=validation_data,
            metric="sharpe",
            number_of_iterations=50,
            max_number_of_features_to_neutralize=120
        )
        print("FINISHED FINDING NEUTRALIZATION FEATURES AND PROPORTIONS FOR BASE MODELS")

        vanilla_predictions = self.data_manager.load_vanila_predictions_data_by_base_models()
        self.model_manager.create_neutralized_predictions_by_base_models_predictions(vanilla_predictions)
        print("FINISHED CREATING NEUTRALIZED PREDICTIONS BY BASE MODELS PREDICTIONS")
