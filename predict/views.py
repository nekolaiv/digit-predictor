from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('predict/digit_model.keras')

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        # Handle image upload
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Process the uploaded image (convert to 28x28 and grayscale)
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        confidence_score = float(np.max(prediction)) * 100  # Convert to percentage
        confidence_score = round(confidence_score, 1) 

        return render(request, 'index.html', {
            'predicted_digit': predicted_digit,
            'confidence_score': confidence_score,
            'image_url': file_url  # Pass the image URL to the template
        })

    return render(request, 'index.html')
