import os
import cv2
import numpy as np
import streamlit as st
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

# ----------------- Utility Functions -----------------
def build_model():
    input_img = Input(shape=(256, 256, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_or_train_model():
    model = build_model()
    if os.path.exists("model/rifd.keras"):
        model.load_weights("model/rifd.keras")
    else:
        if not (os.path.exists('model/X.npy') and os.path.exists('model/Y.npy')):
            st.error("Training data files 'model/X.npy' and/or 'model/Y.npy' not found. Please add them to the 'model' folder.")
            st.stop()
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')

        X = X.astype('float32') / 255.0
        Y = Y.astype('float32') / 255.0

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        checkpoint = ModelCheckpoint('model/rifd.keras', save_best_only=True, verbose=1)
        hist = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                         callbacks=[checkpoint], verbose=1)
        with open('model/rifd.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    return model

def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((256, 256))
    img_np = np.array(img).astype('float32') / 255.0
    return img_np, img

def predict_mask(model, img_np):
    input_arr = np.expand_dims(img_np, axis=0)
    pred = model.predict(input_arr)[0]
    pred = (pred * 255).astype(np.uint8)
    return pred

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Image Forgery Detection", layout="wide")
st.title("🔍 Image Forgery Detection (RIFD-Net)")

uploaded_image = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    img_np, display_img = preprocess_image(uploaded_image)

    with st.spinner("Loading model and predicting..."):
        model = load_or_train_model()
        prediction = predict_mask(model, img_np)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(display_img, caption="Original Image", width=300)

    with col2:
        st.subheader("Forgery Mask")
        st.image(prediction, caption="Predicted Edge Mask", width=300)
