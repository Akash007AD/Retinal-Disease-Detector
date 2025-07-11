import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('eye_cnn.h5')

model = load_model()

# Labels
class_labels = [
    'AMD - Age-related Macular Degeneration',
    'CNV - Choroidal Neovascularization',
    'CSR - Central Serous Retinopathy',
    'DME - Diabetic Macular Edema',
    'DR - Diabetic Retinopathy',
    'DRUSEN - Yellow deposits under the retina',
    'MH - Macular Hole',
    'NORMAL - Healthy eyes with no abnormalities'
]

# UI
st.title("👁️ Retinal Disease Detector")

uploaded_file = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_tensor)
    max_idx = np.argmax(preds)
    predicted = class_labels[max_idx]
    st.success(f"✅ **Predicted disease:** {predicted}")

    # Only call Groq if prediction isn't NORMAL
    if predicted != "NORMAL - Healthy eyes with no abnormalities":
        with st.spinner("Contacting AI doctor..."):
            try:
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are an expert ophthalmologist."},
                        {"role": "user", "content": f"Explain what {predicted} is and possible treatments in under 100 words."}
                    ]
                )
                explanation = completion.choices[0].message.content
                st.info(f"🧬 **Explanation & Treatment:**\n\n{explanation}")
            except Exception as e:
                st.error(f"Error contacting AI doctor: {e}")
    else:
        st.info("🧬 Your eye appears healthy. No disease detected.")
