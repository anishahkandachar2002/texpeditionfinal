import streamlit as st
import os
import requests
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO
from transformers import pipeline, CLIPProcessor, CLIPModel
import torch

# Initialize models
@st.cache_resource
def load_nudity_detection_model():
    class NudityDetection:
        def __init__(self, model_repo='esvinj312/nudity-detection', model_filename='nude_detection_model.h5', target_size=(128, 128)):
            self.model_repo = model_repo
            self.model_filename = model_filename
            self.target_size = target_size
            self.model_path = self.download_model()
            self.model = tf.keras.models.load_model(self.model_path)

        def download_model(self):
            cache_file = f"cached_{self.model_filename}.txt"
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        model_path = f.read().strip()
                        if os.path.exists(model_path):
                            return model_path

                model_path = hf_hub_download(
                    self.model_repo, filename=self.model_filename)

                with open(cache_file, 'w') as f:
                    f.write(model_path)

                return model_path

            except Exception as e:
                raise Exception(f"Failed to download model: {e}")

        def download_image_from_url(self, url):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                raise Exception(f"Failed to download image from URL: {e}")

        def preprocess_image(self, image_path):
            try:
                if isinstance(image_path, Image.Image):
                    img = image_path
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize(self.target_size)
                    img_array = np.array(img) / 255.0
                elif image_path.startswith(('http://', 'https://')):
                    img = self.download_image_from_url(image_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize(self.target_size)
                    img_array = np.array(img) / 255.0
                else:
                    img = load_img(image_path, target_size=self.target_size)
                    img_array = img_to_array(img) / 255.0

                img_array = np.expand_dims(img_array, axis=0)
                return img_array

            except Exception as e:
                raise Exception(f"Could not process image: {e}")

        def predict_image(self, image_path, generate_heatmap=False):
            try:
                img_array = self.preprocess_image(image_path)
                prediction = self.model.predict(img_array)
                percentage_nudity = prediction[0][0] * 100
                is_nsfw = 'NSFW' if percentage_nudity > 50 else 'SFW'
                return is_nsfw, percentage_nudity
            except Exception as e:
                return 'Error', str(e)
    
    return NudityDetection()

@st.cache_resource
def load_toxicity_model():
    return pipeline("text-classification", model="unitary/toxic-bert")

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Initialize all models
nudity_detector = load_nudity_detection_model()
toxicity_pipe = load_toxicity_model()
clip_model, clip_processor = load_clip_model()

# Streamlit UI
st.title("Content Moderation Dashboard")

option = st.sidebar.selectbox(
    "Select Content Type",
    ("Image Analysis", "Text Analysis", "CLIP Content Classification")
)

if option == "Image Analysis":
    st.header("Nudity Detection in Images")
    image_input = st.radio(
        "Select image source",
        ("Upload an image", "Provide image URL")
    )
    
    if image_input == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner('Analyzing image...'):
                    result, confidence = nudity_detector.predict_image(image)
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    col1.metric("Classification", result)
                    col2.metric("Confidence", f"{confidence:.2f}%")
                    
                    if result == "NSFW":
                        st.error("⚠️ This image may contain nudity or explicit content.")
                    else:
                        st.success("✅ This image appears to be safe for work.")
    
    else:
        image_url = st.text_input("Enter image URL:")
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Image from URL', use_column_width=True)
                
                if st.button("Analyze"):
                    with st.spinner('Analyzing image...'):
                        result, confidence = nudity_detector.predict_image(image_url)
                        st.subheader("Results")
                        col1, col2 = st.columns(2)
                        col1.metric("Classification", result)
                        col2.metric("Confidence", f"{confidence:.2f}%")
                        
                        if result == "NSFW":
                            st.error("⚠️ This image may contain nudity or explicit content.")
                        else:
                            st.success("✅ This image appears to be safe for work.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

elif option == "Text Analysis":
    st.header("Toxicity Detection in Text")
    text_input = st.text_area("Enter text to analyze:")
    
    if st.button("Check Toxicity"):
        if text_input:
            with st.spinner('Analyzing text...'):
                result = toxicity_pipe(text_input)[0]
                st.subheader("Results")
                st.metric("Label", result['label'])
                st.metric("Score", f"{result['score']:.4f}")
                
                if result['label'] in ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                    st.error("⚠️ This text contains potentially harmful content.")
                else:
                    st.success("✅ This text appears to be safe.")
        else:
            st.warning("Please enter some text to analyze.")

elif option == "CLIP Content Classification":
    st.header("CLIP Content Classification")
    clip_image_input = st.radio(
        "Select image source for CLIP",
        ("Upload an image", "Provide image URL")
    )
    
    if clip_image_input == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image for CLIP...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image for CLIP', use_column_width=True)
            
            custom_labels = st.text_input("Enter comma-separated labels (optional):", 
                                         "safe for work, not safe for work, contains violence, contains drugs")
            labels = [label.strip() for label in custom_labels.split(",")]
            
            if st.button("Classify with CLIP"):
                with st.spinner('Classifying image...'):
                    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    st.subheader("Classification Probabilities")
                    for label, prob in zip(labels, probs[0]):
                        st.write(f"{label}: {prob.item():.4f}")
                    
                    predicted_label = labels[probs.argmax()]
                    st.metric("Predicted Category", predicted_label)
                    
                    if "not safe" in predicted_label.lower() or "violence" in predicted_label.lower() or "drugs" in predicted_label.lower():
                        st.error("⚠️ This image may contain inappropriate content.")
                    else:
                        st.success("✅ This image appears to be safe.")
    
    else:
        image_url = st.text_input("Enter image URL for CLIP:")
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(image, caption='Image from URL for CLIP', use_column_width=True)
                
                custom_labels = st.text_input("Enter comma-separated labels (optional):", 
                                            "safe for work, not safe for work, contains violence, contains drugs")
                labels = [label.strip() for label in custom_labels.split(",")]
                
                if st.button("Classify with CLIP"):
                    with st.spinner('Classifying image...'):
                        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
                        
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            probs = logits_per_image.softmax(dim=1)
                        
                        st.subheader("Classification Probabilities")
                        for label, prob in zip(labels, probs[0]):
                            st.write(f"{label}: {prob.item():.4f}")
                        
                        predicted_label = labels[probs.argmax()]
                        st.metric("Predicted Category", predicted_label)
                        
                        if "not safe" in predicted_label.lower() or "violence" in predicted_label.lower() or "drugs" in predicted_label.lower():
                            st.error("⚠️ This image may contain inappropriate content.")
                        else:
                            st.success("✅ This image appears to be safe.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses multiple AI models for content moderation:\n"
    "- Nudity detection for images\n"
    "- Toxicity detection for text\n"
    "- CLIP for general content classification"
)