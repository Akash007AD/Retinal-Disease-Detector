# 👁️ Retinal Disease Detector

A deep learning Streamlit web app to detect retinal diseases from OCT images, with AI-generated explanations and treatment suggestions.

✅ Model performance:

* Final test accuracy: \~91.6%
* Trained on \~2800 images across 8 classes
* Custom CNN (5 conv layers + dense layers with batch normalization & dropout)

## Features

* Upload retinal OCT image (jpg, jpeg, png)
* Predicts among 8 retinal conditions:

  * AMD (Age-related Macular Degeneration)
  * CNV (Choroidal Neovascularization)
  * CSR (Central Serous Retinopathy)
  * DME (Diabetic Macular Edema)
  * DR (Diabetic Retinopathy)
  * DRUSEN (Yellow deposits under retina)
  * MH (Macular Hole)
  * NORMAL (Healthy eyes)
* Gets explanation & treatment suggestions from Groq API

## Setup & Installation

Clone this repository:
git clone [https://github.com/your-username/retinal-detector.git](https://github.com/your-username/retinal-detector.git)
cd retinal-detector

Create & activate virtual environment (Python 3.11 recommended):
python -m venv venv

# Windows

venv\Scripts\activate

# Linux/Mac

source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Make sure you have:

* eye\_cnn.h5 file
* .env file with:
  GROQ\_API\_KEY=your\_key\_here

## Run the app

streamlit run app.py

## Tech stack

* Python + Streamlit
* TensorFlow (CNN)
* OpenAI SDK (Groq)
* PIL & NumPy

## Model summary

Final epoch: train accuracy \~93.0%, validation accuracy \~91.7%, val loss \~0.24
Trained over 20 epochs with Adam optimizer and dropout.

## UI

* White & blue gradient background
* Card-like sections for prediction and explanation
* Clean and responsive layout

## Disclaimer

Built for educational use. Not a substitute for professional medical advice.
