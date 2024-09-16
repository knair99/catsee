from catboost import CatBoostClassifier

from resnet_feature_extractor import ResNetWithTransformer


class ImageClassifierWithCatBoost:
    def __init__(self):
        self.feature_extractor = ResNetWithTransformer()
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model('/Users/kprasad/repos/catsee/models/catboost_pet_classifier.cbm')

    def classify(self, image_path):
        features = self.feature_extractor.extract_features(image_path)
        # Classify the image using CatBoost
        prediction = self.catboost_model.predict([features])

        return prediction[0]
