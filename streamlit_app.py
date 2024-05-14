import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
	model = keras.models.load_model('models/20240513-202940.keras')
	return model

def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [256, 256])
	image = np.expand_dims(image, axis = 0)
	prediction = model.predict(image)

	return prediction


def main():
    # model = keras.models.load_model('models/20240513-202940.keras')
    # class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']

    st.title("Tool Classifier")

    st.write("To use this app, upload an image of one of the following tools and click the 'Predict' button.")
    st.write("Combination Wrench, Hammer, Screwdriver, Wrench.")
    st.write("")

    # Add spacing
    st.write("")

    # Load model
    model = load_model()

    # Streamlit image input
    uploaded_file = st.file_uploader(
        label="Upload an image to classify",
        type=['png', 'jpg'],
    )
    
    if uploaded_file is None:
        st.text('Waiting for upload....')

    else:
        slot = st.empty()
        slot.text('Running inference....')
        test_image = Image.open(uploaded_file)
        st.image(test_image, caption="Input Image", width = 400)
        pred = predict_class(np.asarray(test_image), model)
        class_names = ['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']
        result = class_names[np.argmax(pred)]
        output = 'The image is a ' + result
        slot.text('Done')
        st.success(output)

    # if uploaded_file is not None and st.button("Predict"):
    #     # Preprocess the image
    #     image = keras.preprocessing.image.load_img(uploaded_file, target_size=(256, 256))

    #     # Convert the image to a numpy array
    #     input_arr = keras.preprocessing.image.img_to_array(image)

    #     # Normalize the image
    #     input_arr = input_arr / 255.0

    #     # Add a batch dimension
    #     input_arr = np.array([input_arr])

    #     # Make the prediction
    #     prediction = model.predict(input_arr)

    #     # Get the class name
    #     class_index = np.argmax(prediction)
    #     class_name = class_names[class_index]

    #     # Display the class name
    #     st.write(f"The image is of class {class_name}")

    #     # Show bars for the class probabilities
    #     st.bar_chart(pd.Series(prediction[0], index=class_names))

    #     st.success("Prediction done")
      
if __name__=='__main__': 
    main()