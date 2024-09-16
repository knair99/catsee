from flask import Flask, request, jsonify
from models.catboost_classifier import ImageClassifierWithCatBoost
import os  # Import the os module to handle file paths and directory creation

app = Flask(__name__)
classifier = ImageClassifierWithCatBoost()

# Define the upload folder path
UPLOAD_FOLDER = '/Users/kprasad/repos/catsee/static/uploads/'

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename

    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Create the directory if it doesn't exist

    # Save the file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Now classify the image using the saved path
    prediction = classifier.classify(file_path)

    # Return the prediction
    species = "cat" if int(prediction) == 1 else "dog"

    return jsonify({'species': species}), 200

if __name__ == '__main__':
    app.run(debug=True)
