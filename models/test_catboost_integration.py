from catboost_classifier import ImageClassifierWithCatBoost

classifier = ImageClassifierWithCatBoost()
prediction = classifier.classify('/Users/kprasad/repos/catsee/static/sample_images/image.jpg')
print(f'Predicted class: {prediction}')
