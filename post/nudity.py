import os
import requests
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from io import BytesIO


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
            # Check if we have a cached path
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    model_path = f.read().strip()
                    if os.path.exists(model_path):
                        return model_path

            # Download the model
            model_path = hf_hub_download(
                self.model_repo, filename=self.model_filename)

            # Cache the path
            with open(cache_file, 'w') as f:
                f.write(model_path)

            return model_path

        except Exception as e:
            raise Exception(f"Failed to download model: {e}")

    def download_image_from_url(self, url):
        """Download image from URL and return as PIL Image"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            raise Exception(f"Failed to download image from URL: {e}")

    def preprocess_image(self, image_path):
        try:
            # Handle URL vs local path
            if image_path.startswith(('http://', 'https://')):
                # Download image from URL
                img = self.download_image_from_url(image_path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize
                img = img.resize(self.target_size)
                # Convert to array
                img_array = np.array(img) / 255.0
            else:
                # Local file
                img = load_img(image_path, target_size=self.target_size)
                img_array = img_to_array(img) / 255.0

            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        except Exception as e:
            raise Exception(f"Could not process image: {e}")

    def predict_image(self, image_path, generate_heatmap=False, model_path=None):
        try:
            if model_path:
                self.model = tf.keras.models.load_model(model_path)

            img_array = self.preprocess_image(image_path)
            prediction = self.model.predict(img_array)
            percentage_nudity = prediction[0][0] * 100
            is_nsfw = 'NSFW' if percentage_nudity > 50 else 'SFW'

            hm_img = None

            if generate_heatmap:
                hm_img = self.gen_heatmap(image_path)

            return is_nsfw, percentage_nudity, hm_img

        except Exception as e:
            return {'Error': f'Could not process image: {e}'}

    def gen_heatmap(self, image_path):
        try:
            img_array = self.get_img_array(image_path, size=self.target_size)
            heatmap = self.make_gradcam_heatmap(img_array)
            cam_path = self.save_and_display_gradcam(image_path, heatmap)
            return cam_path
        except Exception as e:
            return {'Error': f'Could not generate heatmap: {e}'}

    def get_img_array(self, img_path, size):
        # Handle URL vs local path
        if img_path.startswith(('http://', 'https://')):
            img = self.download_image_from_url(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(size)
            array = np.array(img) / 255.0
        else:
            img = load_img(img_path, target_size=size)
            array = img_to_array(img) / 255.0

        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_array):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(
                "conv2d_2").output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(
            pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def save_and_display_gradcam(self, img_path, heatmap, alpha=0.4):
        # Handle URL vs local path for saving heatmap
        if img_path.startswith(('http://', 'https://')):
            # For URLs, save with a generic name
            img_name = "downloaded_image"
            # Download and save the original image first
            img_pil = self.download_image_from_url(img_path)
            temp_path = f"{img_name}.jpg"
            img_pil.save(temp_path)
            img = cv2.imread(temp_path)
        else:
            img = cv2.imread(img_path)
            img_name = img_path.split('.')[0]

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
        cam_path = f"{img_name}_heatmap.jpg"
        cv2.imwrite(cam_path, superimposed_img)
        return cam_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Usage example
if __name__ == "__main__":
    # Step 1: Create an instance of the NudityDetection class
    detector = NudityDetection()

    # Step 2: Define the path to the image you want to predict
    image_path = input("enter image url")

    # Step 3: Make prediction
    result = detector.predict_image(image_path)

    # Step 4: Print the result
    print(result)

    # Example with heatmap
    # result_with_heatmap = detector.predict_image(image_path, generate_heatmap=True)
    # print(result_with_heatmap)