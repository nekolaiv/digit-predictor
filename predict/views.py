from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import tensorflow as tf
import base64
import pickle
from io import BytesIO

# Load the trained model
with open('models/svm_mnist_model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L")  # Convert to grayscale

    # Auto-invert if background is white
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    # Enhance contrast
    img = ImageOps.autocontrast(img)

    # Resize while maintaining aspect ratio into 20x20
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the image on a black 28x28 canvas
    new_img = Image.new('L', (28, 28), 0)
    left = (28 - img.width) // 2
    top = (28 - img.height) // 2
    new_img.paste(img, (left, top))
    print("image processed")

    return new_img


def index(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        processed_image = preprocess_image(uploaded_file)
        img_array = np.array(processed_image) / 255.0
        img_array = img_array.reshape(1, 784)

        # Predict using SVM
        predicted_digit = int(model.predict(img_array)[0])

        probs = model.predict_proba(img_array)
        confidence_score = round(np.max(probs) * 100, 1)


        return render(request, 'index.html', {
            'predicted_digit': predicted_digit,
            'confidence_score': confidence_score,
            'image_url': file_url
        })

    return render(request, 'index.html')



