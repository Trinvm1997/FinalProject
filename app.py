import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from components import search,features
import os
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
import matplotlib

st.title("Visual Search Engine")
st.header("Cat image classification example")
st.text("Upload a random cat image for searching")

uploaded_file = st.file_uploader("Choose a cat image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded cat', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write("Done.")
    if st.button('Find'):
        st.write("Result...")
        model = features.get_model()

        preprocess = features.get_preprocess_pipeline()

        feature_list = np.load(os.path.join("output/", "features.npy"))
        filename_list = np.load(os.path.join("output/", "filenames.npy"))
        feature_list = feature_list.reshape(9997,2048)

        neighbors = NearestNeighbors(
            n_neighbors=5, algorithm="brute", metric="euclidean"
        ).fit(feature_list)

        im = Image.open(uploaded_file)
        im = preprocess(im)
        im = im.unsqueeze(0)
        with torch.no_grad():
            input_features = model(im).numpy()
        input_features = [input_features.reshape(2048,1).flatten()]

        import time
        tic = time.perf_counter()    
        distances, indices = neighbors.kneighbors([input_features[0]], 5)
        toc = time.perf_counter()
        st.write(f"Search finished in {toc - tic:0.4f} seconds")
        similar_image_paths = filename_list[indices[0]]

        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
        for idx, path in enumerate(similar_image_paths):
            # path = path.replace("/","\\")
            im = Image.open("data\cat\cat\\00000001_000.jpg")
            ax.ravel()[idx].imshow(np.asarray(im))
            ax.ravel()[idx].set_axis_off()
        plt.tight_layout()
        fig.savefig("output/result.png")
        st.write("Done.")
        st.image(Image.open("output/result.png"),caption='Similar cats', use_column_width=True)
