import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from components import search,features
import os, torch, matplotlib, time, joblib, SessionState
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import streamlit.components.v1 as components

matplotlib.use('Agg')
np.random.seed(0)
session_state = SessionState.get(image='', pca='', list='')

def find(uploaded_file):
    model = features.get_model()
    preprocess = features.get_preprocess_pipeline()

    feature_list = np.load(os.path.join("output/", "features.npy"))
    filename_list = np.load(os.path.join("output/", "filenames.npy"))
    feature_list = feature_list.reshape(9997,2048)

    im = Image.open(uploaded_file)
    im = preprocess(im)
    im = im.unsqueeze(0)
    with torch.no_grad():
        input_features = model(im).numpy()
        input_features = [input_features.reshape(2048,1).flatten()]

    n_components = 128
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(feature_list)
    joblib.dump(pca, os.path.join("pca.joblib"))

    feature_length = n_components
    index = AnnoyIndex(feature_length, 'angular')
    for i, j in enumerate(components):
        index.add_item(i, j)

    index.build(15, n_jobs=-1)
    index.save(os.path.join("output/", "index.annoy"))

    pca = joblib.load(os.path.join("output/", "pca.joblib"))
    components = pca.transform(input_features)[0]

    ann_index = AnnoyIndex(components.shape[0], 'angular')
    ann_index.load(os.path.join("output/", "index.annoy"))

    tic = time.perf_counter()
    indices = ann_index.get_nns_by_vector(components, 5, search_k=-1, include_distances=False)
    toc = time.perf_counter()
    st.write(f"Search finished in {toc - tic:0.4f} seconds")
    similar_image_paths = filename_list[indices]

    cols = st.beta_columns(5)
    count = 0
    for image in similar_image_paths:
        image = image.replace("\\","/")
        if count == 0:
            cols[count].image(image,caption=image[13:-4])
            count+=1
        elif count == 1:
            cols[count].image(image,caption=image[13:-4])
            count+=1
        elif count == 2:
            cols[count].image(image,caption=image[13:-4])
            count+=1
        elif count == 3:
            cols[count].image(image,caption=image[13:-4])
            count+=1
        else:
            cols[count].image(image,caption=image[13:-4])
            count-=4
    
    session_state.image = input_features
    session_state.pca = components
    session_state.list = feature_list

def main():
    menu = ["Home","Visualization","Abyssinian","American Bobtail","American Curl"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Visual Search Engine")
        st.header("Cat image classification example")
        st.subheader("Searching Page")
        st.text("Upload a random cat image for searching")

        uploaded_file = st.file_uploader("Choose a cat image ...", type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded cat', use_column_width=True)
            if st.button('Find'):
                find(uploaded_file)
    
    elif choice == "Visualization":
        image = session_state.image
        pca = session_state.pca
        features_list = session_state.list
        cols = st.beta_columns(3)
        cols[0].dataframe(image[0])
        cols[1].text("")
        cols[1].text("")
        cols[1].text("")
        cols[1].text("")
        cols[1].text("")
        cols[1].text("")
        cols[1].text("")
        cols[1].markdown("<h1 style='display: flex; justify-content: center; align-items: center; color: red;'> ===> </h1>", unsafe_allow_html=True)
        cols[2].dataframe(pca)
        st.dataframe(features_list[0:100])

    elif choice == "Abyssinian":
        st.subheader("List of Abyssinian cats")
        filenames = os.listdir("data/images/Abyssinian")
        cols = st.beta_columns(4)
        count=0
        for idx,file in enumerate(filenames):
            if count == 0:
                cols[count].image("data/images/Abyssinian/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 1:
                cols[count].image("data/images/Abyssinian/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 2:
                cols[count].image("data/images/Abyssinian/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            else:
                cols[count].image("data/images/Abyssinian/" + file, caption="Abyssian "+file[0:-4])
                count-=3

    elif choice == "American Bobtail":
        st.subheader("List of American Bobtail cats")
        filenames = os.listdir("data/images/American Bobtail")
        cols = st.beta_columns(4)
        count=0
        for idx,file in enumerate(filenames):
            if count == 0:
                cols[count].image("data/images/American Bobtail/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 1:
                cols[count].image("data/images/American Bobtail/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 2:
                cols[count].image("data/images/American Bobtail/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            else:
                cols[count].image("data/images/American Bobtail/" + file, caption="Abyssian "+file[0:-4])
                count-=3

    else:
        st.subheader("List of American Curl cats")
        filenames = os.listdir("data/images/American Curl")
        cols = st.beta_columns(4)
        count=0
        for idx,file in enumerate(filenames):
            if count == 0:
                cols[count].image("data/images/American Curl/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 1:
                cols[count].image("data/images/American Curl/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 2:
                cols[count].image("data/images/American Curl/" + file, caption="Abyssian "+file[0:-4])
                count+=1
            else:
                cols[count].image("data/images/American Curl/" + file, caption="Abyssian "+file[0:-4])
                count-=3

if __name__ == '__main__':
    main()