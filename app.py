import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model = tf.keras.models.load_model("celebrity_model.keras")


with open("class_names.json", "r") as f:
    class_names = json.load(f)


base_model = model.layers[0]

feature_model = tf.keras.Sequential(
    model.layers[:-1]
)


st.title("Celebrity Face Classifier")

st.write("Upload an image to identify the celebrity.")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    display_image = tf.keras.utils.load_img(
        uploaded_file
    )

    model_image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(
        model_image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(
        image
    )


    prediction = model.predict(
        image,
        verbose=0
    )


    top_3_indices = np.argsort(
        prediction[0]
    )[-3:][::-1]


    class_index = top_3_indices[0]

    celebrity = class_names[class_index]

    confidence = prediction[0][class_index] * 100


    st.success(
        f"Predicted Celebrity: {celebrity}"
    )


    st.subheader("Top 3 Predictions")


    for i in top_3_indices:

        celebrity_name = class_names[i]

        confidence = prediction[0][i] * 100

        st.write(
            f"{celebrity_name}: {confidence:.2f}%"
        )


    dataset_path = "combined"

    celebrity_folder = os.path.join(
        dataset_path,
        celebrity
    )


    if not os.path.exists(
        celebrity_folder
    ):

        st.warning(
            f"Could not find folder for {celebrity}"
        )


    else:

        image_files = []


        for filename in os.listdir(
            celebrity_folder
        ):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_files.append(
                    filename
                )


        best_distance = float("inf")

        best_image_path = None


        query_feature = feature_model.predict(
            image,
            verbose=0
        )


        query_feature = (
            query_feature /
            np.linalg.norm(
                query_feature,
                axis=1,
                keepdims=True
            )
        )


        for filename in image_files:

            image_path = os.path.join(
                celebrity_folder,
                filename
            )


            reference_image = (
                tf.keras.utils.load_img(
                    image_path,
                    target_size=(224, 224)
                )
            )


            reference_image = (
                tf.keras.utils.img_to_array(
                    reference_image
                )
            )


            reference_image = np.expand_dims(
                reference_image,
                axis=0
            )


            reference_image = preprocess_input(
                reference_image
            )


            reference_feature = (
                feature_model.predict(
                    reference_image,
                    verbose=0
                )
            )


            reference_feature = (
                reference_feature /
                np.linalg.norm(
                    reference_feature,
                    axis=1,
                    keepdims=True
                )
            )


            distance = np.linalg.norm(
                query_feature -
                reference_feature
            )


            if distance < best_distance:

                best_distance = distance

                best_image_path = image_path


        if best_image_path is not None:

            best_image = (
                tf.keras.utils.load_img(
                    best_image_path
                )
            )


            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "Your Image"
                )

                st.image(
                    display_image,
                    use_container_width=True
                )


            with col2:

                st.subheader(
                    f"Predicted: {celebrity}"
                )

                st.image(
                    best_image,
                    use_container_width=True
                )


            st.write(
                f"Distance: {best_distance:.4f}"
            )