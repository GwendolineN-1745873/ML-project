from flask import Flask, render_template, request
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    
    img = Image.open(file)
    img = img.resize((224, 224))  # Assuming your model requires input of size 224x224
    img = np.array(img) / 255.0   # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction using your Teras model
    predictions = 5
    
    # Assuming teras.predict() returns a dictionary with classes and probabilities
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
