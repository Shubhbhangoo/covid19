import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained CNN model
model = tf.keras.models.load_model('model.hdf5 ')

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((50, 50)) # Assuming your model expects input size (224, 224)
    img_array = np.array(img) /50.0 # Normalize pixel values
    return img_array.reshape((1, 50, 50, 3)) # Reshape and return

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title('COVID-19 Detection from Chest X-rays')
    st.write('Upload a chest X-ray image and click on "Predict"')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(uploaded_file)

        if prediction[0][0] > 0.5:
            st.write('Prediction: COVID-19 Positive')
        else:
            st.write('Prediction: COVID-19 Negative')

if __name__ == '__main__':
    main()
