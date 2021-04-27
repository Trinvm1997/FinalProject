import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from components import search,features
import os, torch, matplotlib, time, joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import streamlit.components.v1 as components

matplotlib.use('Agg')
np.random.seed(0)

def find(uploaded_file):
    model = features.get_model()
    preprocess = features.get_preprocess_pipeline()

    feature_list = np.load(os.path.join("output/", "features.npy"))
    filename_list = np.load(os.path.join("output/", "filenames.npy"))
    feature_list = feature_list.reshape(32809,2048)

    # neighbors = NearestNeighbors(
    #     n_neighbors=5, algorithm="brute", metric="euclidean"
    # ).fit(feature_list)

    im = Image.open(uploaded_file)
    im = preprocess(im)
    im = im.unsqueeze(0)
    with torch.no_grad():
        input_features = model(im).numpy()
        input_features = [input_features.reshape(2048,1).flatten()]

    # tic = time.perf_counter()    
    # distances, indices = neighbors.kneighbors([input_features[0]], 5)
    # toc = time.perf_counter()
    # st.write(f"Search finished in {toc - tic:0.4f} seconds")
    # similar_image_paths = filename_list[indices[0]]

    n_components = 128
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(feature_list)
    joblib.dump(pca, os.path.join("pca.joblib"))

    feature_length = n_components
    index = AnnoyIndex(feature_length, 'angular')
    for i, j in enumerate(components):
        index.add_item(i, j)

    index.build(15)
    index.save(os.path.join("output/", "index.annoy"))

    pca = joblib.load(os.path.join("output/", "pca.joblib"))
    components = pca.transform(input_features)[0]

    ann_index = AnnoyIndex(components.shape[0], 'angular')
    ann_index.load(os.path.join("output/", "index.annoy"))

    tic = time.perf_counter()
    indices = ann_index.get_nns_by_vector(components, 5, search_k=-1, include_distances=False)
    indices = np.array(indices)
    toc = time.perf_counter()
    st.write(f"Search finished in {toc - tic:0.4f} seconds")
    similar_image_paths = filename_list[indices]

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
    for idx, path in enumerate(similar_image_paths):
        im = Image.open(path.replace("\\","/"))
        ax.ravel()[idx].imshow(np.asarray(im))
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    fig.savefig("result.png")

def main():
    menu = ["Home","About","Abyssinian","American Bobtail","American Curl"]
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
<<<<<<< Updated upstream
                find(uploaded_file)
=======
                model = features.get_model()
                preprocess = features.get_preprocess_pipeline()

                feature_list = np.load(os.path.join("output/", "features.npy"))
                filename_list = np.load(os.path.join("output/", "filenames.npy"))
                feature_list = feature_list.reshape(9997,2048)

                # neighbors = NearestNeighbors(
                #     n_neighbors=5, algorithm="brute", metric="euclidean"
                # ).fit(feature_list)

                im = Image.open(uploaded_file)
                im = preprocess(im)
                im = im.unsqueeze(0)
                with torch.no_grad():
                    input_features = model(im).numpy()
                    input_features = [input_features.reshape(2048,1).flatten()]

                # tic = time.perf_counter()    
                # distances, indices = neighbors.kneighbors([input_features[0]], 5)
                # toc = time.perf_counter()
                # st.write(f"Search finished in {toc - tic:0.4f} seconds")
                # similar_image_paths = filename_list[indices[0]]

                n_components = 128
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(feature_list)
                joblib.dump(pca, os.path.join("pca.joblib"))

                feature_length = n_components
                index = AnnoyIndex(feature_length, 'angular')
                for i, j in enumerate(components):
                    index.add_item(i, j)

                index.build(15)
                index.save(os.path.join("output/", "index.annoy"))

                pca = joblib.load(os.path.join("output/", "pca.joblib"))
                components = pca.transform(input_features)[0]

                ann_index = AnnoyIndex(components.shape[0], 'angular')
                ann_index.load(os.path.join("output/", "index.annoy"))

                tic = time.perf_counter()
                indices = ann_index.get_nns_by_vector(components, 5, search_k=-1, include_distances=False)
                indices = np.array(indices)
                toc = time.perf_counter()
                st.write(f"Search finished in {toc - tic:0.4f} seconds")
                similar_image_paths = filename_list[indices]
                st.text(similar_image_paths)

                fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
                for idx, path in enumerate(similar_image_paths):
                    im = Image.open(path.replace("\\","/"))
                    ax.ravel()[idx].imshow(np.asarray(im))
                    ax.ravel()[idx].set_axis_off()
                plt.tight_layout()
                fig.savefig("result.png")
>>>>>>> Stashed changes
                st.image(Image.open("result.png"),caption='Similar cats', use_column_width=True)
    
    elif choice == "About":
        st.subheader("About App")
        st.text("This App is designed to stimulate")
        st.text("how a visual search engine is implemented into an E-Commerce website")
        st.text("the souce code and document can be found at")

    elif choice == "Abyssinian":
        st.subheader("List of Abyssinian cats")
        filenames = os.listdir("data\images\Abyssinian")
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
        filenames = os.listdir("data\images\American Bobtail")
        cols = st.beta_columns(4)
        count=0
        for idx,file in enumerate(filenames):
            if count == 0:
                cols[count].image("data/images/American Bobtail/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 1:
                cols[count].image("data/images/American Bobtail/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 2:
                cols[count].image("data/images/American Bobtail/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            else:
                cols[count].image("data/images/American Bobtail/" + file, width=150, caption="Abyssian "+file[0:-4])
                count-=3

    else:
        st.subheader("List of American Curl cats")
        filenames = os.listdir("data\images\American Curl")
        cols = st.beta_columns(4)
        count=0
        for idx,file in enumerate(filenames):
            if count == 0:
                cols[count].image("data/images/American Curl/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 1:
                cols[count].image("data/images/American Curl/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            elif count == 2:
                cols[count].image("data/images/American Curl/" + file, width=150, caption="Abyssian "+file[0:-4])
                count+=1
            else:
                cols[count].image("data/images/American Curl/" + file, width=150, caption="Abyssian "+file[0:-4])
                count-=3

if __name__ == '__main__':
    main()