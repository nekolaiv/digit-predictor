# 🧠 Digit Predictor — Project Documentation

## 📌 Overview

**Digit Predictor** is a machine learning web application developed using **Django** that predicts handwritten digits (0–9) from uploaded images. It uses a Support Vector Machine (SVM) classifier trained on the MNIST dataset. The project includes full-stack functionality with authentication, prediction dashboard, data storage, and visualization.

---

## 📊 Dataset Description

**Dataset Used**: [MNIST Handwritten Digit Dataset](http://yann.lecun.com/exdb/mnist/)

* **Source**: `sklearn.datasets.fetch_openml('mnist_784')`
* **Size**: 70,000 grayscale images
* **Dimensions**: 28 x 28 pixels
* **Classes**: 10 (digits 0 through 9)
* **Shape**: Each image is represented as a flattened 784-length vector (28×28)

---

## 🔁 ML Model Training Steps

### 1. **Loading the Dataset**

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
```

### 2. **Preprocessing**

* Normalize pixel values from \[0–255] to \[0–1] by dividing by 255.0
* Convert labels from strings to integers
* Flatten 28x28 images into 784-dimensional vectors

```python
X = X / 255.0
y = y.astype('int')
```

### 3. **Splitting the Data**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4. **Training the Model**

```python
from sklearn.svm import SVC

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
```

### 5. **Saving the Model**

```python
import pickle

with open("digit_predictor_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

---

## 🔐 How Authentication Was Added

### Django’s Built-in Authentication System

* `django.contrib.auth` used for login/logout/session handling
* `User` model extended to manage user accounts

### Features Implemented:

* **Login Page**: Styled with TailwindCSS, handles errors and redirects
* **Reset Password**:

  * Users enter username and new password
  * Password validation includes:

    * Preventing reuse of the old password
    * Password confirmation match
* **Conditional Access**:

  * Redirects logged-in users away from login page
  * Protects dashboard and prediction pages using `@login_required`

---

## 🔌 Steps for Integration

### 1. **Prediction View**

```python
@csrf_exempt
@require_POST
def predict_digit(request):
    base64_image = request.POST.get("image")
    processed_image = preprocess_image(base64_image)
    img_array = np.array(processed_image).reshape(1, 784) / 255.0

    with open("digit_predictor_model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = int(model.predict(img_array)[0])
    confidence = round(model.predict_proba(img_array).max() * 100, 2)

    Prediction.objects.create(digit=prediction, confidence=confidence)

    return JsonResponse({"prediction": prediction, "confidence": confidence})
```

### 2. **Frontend Integration**

* JavaScript used to capture image input from canvas
* Image sent via `fetch` to `/predict/` endpoint as base64
* AJAX used to preview post-processed image and return prediction

### 3. **Database Integration**

* Prediction model stores:

  * Digit
  * Confidence score
  * Timestamp
* Stored using Django’s ORM into PostgreSQL

---

## ⚠️ Challenges Encountered

| Challenge                                           | Solution                                                                                |
| --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Handling base64 canvas image uploads                | Wrote a custom `preprocess_image()` to decode, resize, grayscale, and flatten the image |
| Preventing repeat password use                      | Validated with `user.check_password(new_password)`                                      |
| Avoiding duplicate login when already authenticated | Used `request.user.is_authenticated` check to redirect                                  |
| Incorrect predictions due to poor image quality     | Applied normalization and image sharpening during preprocessing                         |
| Model not saving predictions                        | Ensured model call and `Prediction.objects.create()` executed after base64 decode       |
| Login page accessible even after login              | Added conditional redirect if user is already logged in                                 |

---

## ✅ Final Remarks

This project demonstrates end-to-end integration of **machine learning**, **image processing**, **web development**, and **user authentication**. It provides a user-friendly and intuitive interface for digit recognition and lays the groundwork for further applications in optical character recognition (OCR) and AI-driven educational tools.
