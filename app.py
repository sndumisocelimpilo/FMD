import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

st.set_page_config(page_title="FMD Detector", layout="wide")
st.title("🐄 FMD Detector – Eswatini")
st.markdown("Upload a photo of your cow to check for Foot and Mouth Disease.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Original", use_container_width=True)
    with st.spinner("Analysing..."):
        prob = predict(preprocess(img))
        pred = 1 if prob > 0.5 else 0
        conf = prob if pred == 1 else 1 - prob
        result = "FMD" if pred == 1 else "Healthy"
        st.success(f"**Diagnosis:** {result} (confidence: {conf:.1%})")
        if pred == 1:
            st.warning("Isolate the animal. Disinfect equipment. Call a vet immediately.")
        else:
            st.info("No signs of FMD. Continue routine monitoring.")
        if st.button("📞 Call Veterinary Officer"):
            st.info("Call the nearest extension officer: +268 1234 5678")
