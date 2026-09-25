import streamlit as st
import tensorflow as tf 
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="Solar Panel Defect Detection", page_icon=":camera:", layout="centered")
st.title(":sun_with_face: Solar Panel Defect Detection")
st.write("Upload an image of a solar panel to detect defects using a deep learning model.")

with st.sidebar:
    st.header("About")
    st.info("""
        This app uses a deep learning model(e.g., CNN based on ResNet/EfficientNet) trained on public solar panel defect datasets.
        Common detected classess includes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-Covered
        """)

# ----------------------------------------Class Names ----------------------------------
CLASSES =["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-damage", "Snow-Covered",
          "Cell","Cell-Multi","Cracking","Hot-Spot","Hot-Spot-Multi", "Shadowing","Diode",
          "Diode-Multi","Vegetation","Soiling","Offline","No-Anomaly"]

# ----------------------------------------Load Model(Cached) -----------------------------------

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_effenet_finetune.keras")
    return model

with st.spinner("Loading model..."):
    model = load_model()


    # File uploader for image input
    uploaded_files = st.file_uploader("Upload a solar panel image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    # Process the batch if files exist
    if uploaded_files:
        st.success(f"Successfully loaded{len(uploaded_files)}files. Processing matrix...")
        # Store results to display later in a table or map
        results_summary =[]

        for uploaded_file in uploaded_files:
        # Open the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image for prediction
            img = image.resize((224, 224))
            img_array = np.array(img)  # Normalize pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array.astype(np.float32))  # Preprocess for EfficientNet

            with st.spinner("Analysing the panel..."):

            # Make Model Prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                confidence_score = predictions[0][predicted_index]

            predicted_class = CLASSES[predicted_index]
            # Append report metrics
            results_summary.append({ "filename": uploaded_file.name,"predicted_class": predicted_class,"confidence_score": float(confidence_score)})


            st.markdown(f"### Predicted Class: **{predicted_class}** ***Confidence Score: {confidence_score:.1%}**")

            if predicted_class == "Clean":
                st.success("The solar panel is clean and good condition.")
            else:
                st.warning(f"The solar panel has a defect: **{predicted_class}**. Please inspect and take necessary action.")

        # Top 3 Predictions
            st.write("### Top 3 Predictions:")
            top_indices = np.argsort(predictions[0])[-3:][::-1]  # Get indices of top 3 predictions in descending order
            for i, idx in enumerate(top_indices):
                class_name = CLASSES[idx]
                prob = predictions[0][idx]


            # Color code: green for 1st, blue for 2nd and orange for 3rd
                if i == 0:
                    st.markdown(f"**1st:** {class_name} - {prob:.1%}")
                elif i == 1:
                    st.markdown(f"**2nd:** {class_name} - {prob:.1%}")
                else:
                    st.markdown(f"**3rd:** {class_name} - {prob:.1%}")


            with st.expander("View all class  Probabilities"):
                for i, prob in enumerate(predictions[0]):
                    st.write(f"{CLASSES[i]:<20}{prob:.1%}")
            
        
