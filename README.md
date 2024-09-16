# Weekend Project Plan: Build an Image Classification API with Transformer and CatBoost
This project expands your image classification pipeline by adding CatBoost as a final decision-making step, with ResNet and a Transformer for feature extraction. This mimics the structure of your Image Automod project but keeps the task manageable for a weekend timeline.


## Plan
0. We will first implement ResNet pretrained model along with a CatBoost Classifier
Purpose: Build a pipeline that processes images using a pre-trained ResNet model for feature extraction, adds a transformer for refined features, and uses CatBoost for multiclass classification. This simulates how your Image Automod service leverages multiple models.
1. Set Up ResNet for Feature Extraction
2. Train a Simple CatBoost Model and save it
3. Test the Full Pipeline by Classifying a Sample Image
4. Build Flask API
5. Test Flask API with a Sample Image and have the model predict the class
5. Integrate Transformer: Add a transformer layer to enhance feature extraction before passing the features to the CatBoost model.
-- Modify the ImageClassifierWithCatBoost class to use this enhanced transformer-based feature extractor.
6. Final Test

## API Routes

```aiignore
@app.route('/classify', methods=['POST'])

```
## Test API Using Postman:

1. Use Postman to send a POST request with an image file to /classify
2. Test the API with Postman after adding the transformer to ensure everything works end-to-end.
This plan gives you a clear step-by-step guide to build the system over the weekend while staying focused on your Monday deadline. Let me know if you need any further clarification as you work through it!


## Folder structure
```aiignore
image-classification-api/
├── api/
│   └── app.py (Flask API logic)
├── models/
│   └── classifier.py (ResNet + Transformer + CatBoost)
├── static/
│   └── sample_images/ (test images)
└── README.md
Test Flask Setup: Write a simple "Hello World" Flask app to verify that everything is running:
```