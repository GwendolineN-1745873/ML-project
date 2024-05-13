import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb')) 
cols=['age','workclass','education','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss',
      'hours-per-week','native-country']    
  
def main(): 
    st.title("Tool Classifier")
    html_temp = """
    <div style="background:#efeee4 ;padding:10px">
    <h2 style="color:white;text-align:center;">Tool Classifier </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    # Streamlit image input
    image_to_classify = st.file_uploader(
        label="Upload an image to classify",
        type=['png', 'jpg'],

    )

    # if uploaded_file is not None:
    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

    if uploaded_file is not None and st.button("Predict"):
        # Rescale and resize image
        image = Image.open(uploaded_file)
        prediction = model.predict(image)

        st.success()
      
if __name__=='__main__': 
    main()