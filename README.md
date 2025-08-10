# 🐟 Fish Species Classifier

A deep learning project that classifies fish species from images using TensorFlow/Keras and a Streamlit web app.

---

## 📌 Features
- Trains and evaluates multiple models (including CNN from scratch and transfer learning models).
- Displays accuracy, loss, and predictions in an interactive Streamlit interface.
- Allows users to upload an image and get predictions instantly.

---

## 📂 Project Structure
```bash
├── app.py
├── fishclassification.ipnyb
├── requirements.txt
├── README.md
└── cnn_best.h5
```

---

## 📊 Dataset
The dataset used for training is available here:  
[📂 Download Dataset](https://drive.google.com/drive/folders/1RngjGmKV4b4PJj2FZvOe5MOAeSS3hkMK?usp=sharing)  

After downloading:
1. Extract it into the project folder.
2. Ensure the structure matches your data generator settings (e.g., `train/`, `valid/`, `test/`).

---

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/Alfiya-Simran/fish-classification
cd fish-classification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App
```bash
streamlit run app.py
```

---

## 📷 Example Prediction

---

## 🧠 Models Used
1. CNN from Scratch

2. Transfer Learning with:

3. MobileNetV2

4. InceptionV3

5. ResNet50

6. EfficientNetB0

## 📄 License
This project is licensed under the MIT License.
