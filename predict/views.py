from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import pickle
import base64
from io import BytesIO
from .models import Prediction
from django.db.models import Count, Avg
from django.utils.timezone import now, timedelta
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from io import BytesIO
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required


# Load the trained model
with open('models/svm_mnist_model.pkl', 'rb') as file:
    model = pickle.load(file)



def preprocess_image(image_input):
    if isinstance(image_input, str) and image_input.startswith('data:image'):
        # Base64 case
        header, encoded = image_input.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_file = BytesIO(image_data)
    else:
        image_file = image_input  # File upload

    img = Image.open(image_file).convert("L")  # Grayscale

    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    img = ImageOps.autocontrast(img)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), 0)
    left = (28 - img.width) // 2
    top = (28 - img.height) // 2
    new_img.paste(img, (left, top))

    return new_img


@csrf_exempt
@require_POST
def predict_digit(request):
    base64_image = request.POST.get("image")
    if base64_image:
        processed_image = preprocess_image(base64_image)
        img_array = np.array(processed_image) / 255.0
        img_array = img_array.reshape(1, 784)

        predicted_digit = int(model.predict(img_array)[0])
        if hasattr(model, "predict_proba"):
            confidence_score = round(float(model.predict_proba(img_array).max()) * 100, 2)
        else:
            confidence_score = 0.0

        # Save to DB
        try:
            Prediction.objects.create(digit=predicted_digit, confidence=confidence_score)
        except Exception as e:
            print(f"DB save failed: {e}")


        return JsonResponse({
            "prediction": predicted_digit,
            "confidence": confidence_score,
        })

    return JsonResponse({"error": "No image provided."}, status=400)

@login_required(login_url='login')
def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        processed_image = preprocess_image(uploaded_file)

        img_array = np.array(processed_image) / 255.0
        img_array = img_array.reshape(1, 784)

        predicted_digit = int(model.predict(img_array)[0])
        probs = model.predict_proba(img_array)
        confidence_score = round(np.max(probs) * 100, 1)

        buffer = BytesIO()
        processed_image.save(buffer, format='PNG')
        buffer.seek(0)
        content_file = ContentFile(buffer.read(), 'processed.png')

        fs = FileSystemStorage()
        processed_file_name = fs.save('processed.png', content_file)
        processed_file_url = fs.url(processed_file_name)

        Prediction.objects.create(digit=predicted_digit, confidence=confidence_score)

        return render(request, 'home.html', {
            'predicted_digit': predicted_digit,
            'confidence_score': confidence_score,
            'image_url': processed_file_url  # Use processed image in template
        })

    return render(request, 'home.html')




def index(request):
    return render(request, 'index.html')


@login_required(login_url='login')
def dashboard_view(request):
    total = Prediction.objects.count()
    avg_conf = Prediction.objects.aggregate(avg=Avg('confidence'))['avg'] or 0
    digit_counts = Prediction.objects.values('digit').annotate(c=Count('id'))
    most_pred = max(digit_counts, key=lambda x: x['c'])['digit'] if digit_counts else None
    least_pred = min(digit_counts, key=lambda x: x['c'])['digit'] if digit_counts else None

    # Bar chart
    labels = list(range(10))
    counts = [next((d['c'] for d in digit_counts if d['digit'] == i), 0) for i in labels]

    # Line chart (last 7 days)
    days = [(now() - timedelta(days=i)).date() for i in reversed(range(7))]
    timeline_labels = [d.strftime("%b %d") for d in days]
    timeline_counts = [Prediction.objects.filter(timestamp__date=d).count() for d in days]

    # Confidence histogram (10% bins)
    bins = list(range(0, 101, 10))
    hist = [Prediction.objects.filter(confidence__gte=b, confidence__lt=b+10).count() for b in bins]

    context = {
        'total_predictions': total,
        'avg_confidence': round(avg_conf, 2),
        'most_predicted_digit': most_pred,
        'least_predicted_digit': least_pred,
        'digit_labels': labels,
        'digit_counts': counts,
        'timeline_labels': timeline_labels,
        'timeline_counts': timeline_counts,
        'confidence_bins': [f"{b}-{b+10}%" for b in bins],
        'confidence_values': hist,
    }
    return render(request, 'dashboard.html', context)


