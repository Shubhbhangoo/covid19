import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Assuming TensorFlow backend

# Load your pre-trained CNN model
model = load_model('model.h5')  # Replace with your model filename

def preprocess_image(image):
    """Preprocesses an X-ray image for CNN input."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)  # Add a batch dimension

def predict(image):
    """Predicts the presence of COVID-19 in an X-ray image."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)  # Get the index of the predicted class
    class_names = ['Negative', 'COVID-19']  # Replace with your class names
    return class_names[class_index]

# Streamlit app setup
st.title('COVID-19 Detection from X-ray Images')
st.write('Upload an X-ray image to check for potential COVID-19 presence.')
uploaded_file = st.file_uploader("Choose an X-ray image:", type="jpg,png")

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels='BGR')

    if st.button('Predict'):
        prediction = predict(image)
        st.write(f'Prediction: {prediction}')

        if prediction == 'COVID-19':
            st.warning('This is a suggestion only. Consult a medical professional for diagnosis.')
        else:
            st.success('The X-ray appears negative for COVID-19.')
