# Digit Predictor Project

## Overview

This project is a **Digit Predictor** application designed to predict handwritten digits using machine learning. It leverages a **Support Vector Machine (SVM)** classifier trained on the **MNIST** dataset, which contains images of handwritten digits from 0 to 9. The application allows users to upload an image of a handwritten digit, preprocess the image, and predict the corresponding digit along with a confidence score. The project uses **Django** for the web framework, and the image preprocessing and prediction are handled via a Python-based backend.

## Table of Contents

* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model and Preprocessing](#model-and-preprocessing)
* [Training Process](#training-process)
* [License](#license)
* [Contributing](#contributing)

## Features

* **Image Upload**: Users can upload images of handwritten digits.
* **Digit Prediction**: The model predicts the digit represented by the uploaded image.
* **Confidence Score**: Displays the confidence score for the predicted digit, showing how confident the model is in its prediction.
* **Image Preprocessing**: The uploaded image is processed to match the input format expected by the model.
* **Results Display**: The prediction and confidence score are displayed alongside the uploaded image for user verification.
* **Dashboard**: A dashboard interface that shows the result, history of predictions, and easy access to the prediction tool.

## Technologies Used

* **Django**: Web framework used for building the application.
* **Python 3.x**: Backend programming language.
* **scikit-learn**: Used for training the machine learning model and predicting the digit.
* **NumPy**: For numerical data manipulation and processing.
* **Pandas**: For data handling and manipulation.
* **Matplotlib**: For data visualization, especially during model training.
* **HTML/CSS (TailwindCSS)**: Frontend styling for the web interface.
* **JavaScript**: For dynamic content such as AJAX and handling image previews.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-nekolaiv/digit-predictor.git
cd digit-predictor
```

### Step 2: Set up a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up Database (if using Django's default database)

Run the migrations to set up the database:

```bash
python manage.py migrate
```

### Step 5: Run the Development Server

```bash
python manage.py runserver
```

Visit the application at `http://127.0.0.1:8000` in your browser.

## Usage

1. **Upload an Image**: On the homepage, click the "Browse" button to upload an image of a handwritten digit.
2. **Process the Image**: After uploading, the image will be preprocessed and passed to the machine learning model for prediction.
3. **View Prediction**: The predicted digit and its confidence score will be displayed alongside the original image.
4. **View History**: The application can save the prediction history (digit and confidence score), which can be accessed via the dashboard.

## Model and Preprocessing

### Image Preprocessing

To ensure the image is in the correct format for prediction, it is preprocessed as follows:

1. **Resize**: The image is resized to 28x28 pixels, the standard input size for the MNIST dataset.
2. **Grayscale Conversion**: The image is converted to grayscale (if it's a color image), as the model was trained on grayscale images.
3. **Normalization**: The pixel values of the image are normalized to a range of 0 to 1 by dividing each pixel value by 255.0.
4. **Flattening**: The image is flattened into a 1D array of 784 pixels (28x28 = 784), which is the input format expected by the SVM model.

### Model Training

The model is based on the **Support Vector Machine (SVM)** classifier from scikit-learn. The SVM model is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits.

1. **Dataset**: The MNIST dataset is a collection of grayscale images of handwritten digits (0-9), each 28x28 pixels in size.
2. **Preprocessing for Training**: The dataset is preprocessed similarly to the input images (grayscale conversion, normalization, and flattening).
3. **Model**: The model uses an SVM classifier with a linear kernel, which works well for this type of classification problem.
4. **Training**: The SVM model is trained using the `scikit-learn` library with the training data from the MNIST dataset.

After training, the model is saved to a file (using `joblib`) for later use in the application. The trained model is then loaded each time the application runs and used for making predictions.

## Training Process

### Step 1: Load the MNIST Dataset

```python
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X = X / 255.0  # Normalize pixel values
```

### Step 2: Train the SVM Model

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using SVM
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)
```

### Step 3: Save the Trained Model

```python
import pickle

# Save the trained model to a file using pickle
with open("digit_predictor_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

### Step 4: Use the Model for Prediction

In the application, the trained model is loaded and used to make predictions on preprocessed images:

```python
model = joblib.load("digit_predictor_model.joblib")

# Make predictions
predicted_digit = model.predict(image_array)
confidence_score = model.predict_proba(image_array).max()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
