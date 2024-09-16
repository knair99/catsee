from catboost import CatBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from resnet_feature_extractor import retrieve_oxford_feature_dataset

image_features, image_labels = retrieve_oxford_feature_dataset()

X_train, X_test, y_train, y_test = train_test_split(image_features, image_labels, test_size=0.2, random_state=42)

catboost_model = CatBoostClassifier(iterations=1000,
                                    depth=4,
                                    learning_rate=0.1,
                                    loss_function='MultiClass')

# Train the catboost model on the image training dataset
catboost_model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = catboost_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the trained model for later use
catboost_model.save_model('catboost_pet_classifier.cbm')
