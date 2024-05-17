import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
import altair as alt


@st.cache_resource()
def load_model():
	model = keras.models.load_model('model_info/test/model.keras', compile=False)
	return model

def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [256, 256])
	image = np.expand_dims(image, axis = 0)
	prediction = model.__call__(image)
	return prediction


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
    - **Dataset**: Used for training and testing the classifier, consisting of images of various tools.
    - **Design Decisions**:
      - **Model**: Convolutional Neural Network (CNN)
      - **Hyperparameters**: Specific parameters used for training (e.g., learning rate, batch size)
      - **Data Distribution**: Training, validation, and testing splits
      - **Pre-processing**: Resizing images to 256x256 pixels, normalization, and data augmentation techniques.
    """)

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
        top_3_indices = np.argsort(pred[0])[-3:][::-1]
        top_3_predictions = [(class_names[i], pred[0][i] * 100) for i in top_3_indices]
        
        output = f'Top 3 predictions:'
        for i, (class_name, confidence) in enumerate(top_3_predictions):
            output += f'\n{i+1}. {class_name} with {confidence:.2f}% confidence.'
        
        slot.text('Done')
        st.success(output)

        # Get top prediction
        top_index = np.argmax(pred)
        top_class = class_names[top_index]
        confidence = pred[0][top_index] * 100

        # Create a DataFrame for the bar chart
        pred_df = pd.DataFrame({
            'Class': class_names,
            'Confidence': pred[0] * 100
        })

        # Display the bar chart with horizontal labels using Altair
        chart = alt.Chart(pred_df).mark_bar().encode(
            x=alt.X('Class', sort='-y'),
            y=alt.Y('Confidence'),
        ).properties(
            width=600,
            height=400
        ).configure_axis(
            labelAngle=0  # Make x-axis labels horizontal
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

        # Display the top prediction
        output = f'The image is a {top_class} with {confidence:.2f}% confidence.'
        st.success(output)





        # # Get top prediction
        # top_index = np.argmax(pred)
        # top_class = class_names[top_index]
        # confidence = pred[0][top_index] * 100
        
        # # Create a DataFrame for the bar chart
        # pred_df = pd.DataFrame({
        #     'Class': class_names,
        #     'Confidence': pred[0] * 100
        # })
        
        # # Create the bar chart using Altair
        # bars = alt.Chart(pred_df).mark_bar(color='yellow').encode(
        #     x=alt.X('Class', sort=None),
        #     y=alt.Y('Confidence', scale=alt.Scale(domain=[0, 100])),
        #     tooltip=['Class', 'Confidence']
        # ).properties(
        #     width=600,
        #     height=400
        # ).configure_axis(
        #     labelAngle=0  # Make x-axis labels horizontal
        # ).configure_view(
        #     strokeWidth=0  # Remove the border around the chart
        # )

        # # Create the text for the bars
        # text = bars.mark_text(
        #     align='center',
        #     baseline='middle',
        #     dy=-10,  # Adjust the position of the text
        #     color='white'  # Make text color white for visibility
        # ).encode(
        #     text='Confidence:Q'
        # )

        # # Combine the bar and text charts
        # chart = bars + text

        # # Display the chart in Streamlit
        # st.altair_chart(chart.configure_axis(labelAngle=0), use_container_width=True)

if __name__ == '__main__':
    main()