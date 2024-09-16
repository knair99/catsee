from catboost import CatBoostClassifier
from resnet_feature_extractor import ResnetFeatureExtractor


class ImageClassifierWithCatBoost:
    def __init__(self):
        self.feature_extractor = ResnetFeatureExtractor()
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model('/Users/kprasad/repos/catsee/models/catboost_model.cbm')

    def classify(self, image_path):
        features = self.feature_extractor.extract_features(image_path)
        return self.catboost_model.predict(features)