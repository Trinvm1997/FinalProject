import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from components import search,features

st.title("Visual Search Engine")
st.header("Cat image classification example")
st.text("Upload a random cat image for image classification")

uploaded_file = st.file_uploader("Choose a cat image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded cat', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.progress(progress_variable_1_to_100)
    st.write("Done.")
    if st.button('Find'):
        st.write("Result...")
        outputImage = search.knn(image, output, features)

    