
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import altair as alt


@st.cache_resource()
def load_model():
	model = keras.models.load_model('model_info/maybethebest/model.keras', compile=False)
	return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    
    # Check if shape is correct (1, 256, 256, 3)
    image = np.expand_dims(image, axis = 0)
    
    # Check if shape is correct (1, 256, 256, 3)
    if (image.shape[3] != 3):
        image = image[..., :3]
        
    print(image.shape)
    
    prediction = model.__call__(image)
    
    return prediction

# Function to open the image when button is clicked
def toggle_image():
    if not st.session_state.is_image_open:
        with open("model.jpg", "rb") as img_file:
            img_bytes = img_file.read()
        st.image(img_bytes, caption="Model Architecture", use_column_width=True)
        st.session_state.is_image_open = True
    else:
        st.session_state.is_image_open = False

# Create a button to toggle the image display
if not hasattr(st.session_state, "is_image_open"):
    st.session_state.is_image_open = False



def main():
    # Load the model
    model = load_model()
    class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

    # Title and description
    st.title("Tool Classifier")
    st.write("**Authors:** Melih Demirel and Gwendoline Nijssen")
    st.write("This app classifies images of tools into one of the following categories: Combination Wrench, Hammer, Screwdriver, and Wrench.")

    st.write("""
    ### Dataset and Model Information
    - **Dataset**: 50 real images used for each category.
    - **Design Decisions**:
      - **Data Distribution**:  70/20/10 split for training, validation and testing
      - **Model**: 
        - **CNN**:
            - Data Augmentation first
            - Filters: 128, 64, 32, 16 and 32 back to back
            - Kernel size: (3, 3)
            - Batch size: 16
      - **Hyperparameters**: Specific parameters used for training (e.g., learning rate, batch size)
      - **Pre-processing**: Resizing images to 256x256 pixels, normalization, and data augmentation techniques (like RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomCrop and RandomContrast).
""")

    if st.button("Click to see model architecture"):
        toggle_image()
    st.write("")

    # File uploader for image input
    uploaded_file = st.file_uploader(
        label="Upload an image to classify",
        type=['png', 'jpg'],
    )

    if uploaded_file is None:
        st.text('Waiting for upload...')
    else:
        slot = st.empty()
        slot.text('Running inference...')
        test_image = Image.open(uploaded_file)
        st.image(test_image, caption="Input Image", width=400)

        pred = predict_class(np.asarray(test_image), model)
        
        # Get top 3 predictions
        predictions = np.argsort(pred[0])[-4:][::-1]
        preds = [(class_names[i], pred[0][i] * 100) for i in predictions]
        
        output = f'Predictions:'
        for i, (class_name, confidence) in enumerate(preds):
            output += f'\n{i+1}. {confidence:.2f}% : {class_name}'
        
        slot.text('Done')
        st.success(output)

        st.image('model.jpg')


if __name__ == '__main__':
    main()
