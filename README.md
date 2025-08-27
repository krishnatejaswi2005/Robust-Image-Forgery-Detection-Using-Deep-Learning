# RIFD-Net: Robust Image Forgery Detection Network

🔍 **RIFD-Net** is a deep learning project designed to detect and localize image forgeries using a convolutional autoencoder.
The project provides a **Streamlit web app** where users can upload images, and the model predicts a **forgery mask** highlighting suspicious/edited regions.

---

## 📌 Features

- Upload any image (JPG/PNG) and get a predicted **forgery mask**.
- Trains a **convolutional autoencoder** for detecting manipulated regions.
- Uses **Keras (TensorFlow backend)** for model building.
- Provides both training and inference pipelines.
- Web UI built with **Streamlit** for an interactive experience.

---

## 🧠 How It Works

1. **Architecture**

   - Encoder–decoder (autoencoder) built with **Conv2D, MaxPooling2D, and UpSampling2D** layers.
   - Learns reconstruction differences to highlight forged areas.

2. **Training Process**

   - Dataset (`X.npy` and `Y.npy`) contains original images (`X`) and corresponding edge/forgery masks (`Y`).
   - Model is trained using **Mean Squared Error (MSE) loss** with **Adam optimizer**.
   - Training checkpoints are saved to `model/rifd.keras`.

3. **Prediction**

   - Input image is preprocessed (resized to `256×256`, normalized).
   - Model outputs a **predicted mask** where tampered regions are emphasized.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Handling:** NumPy, OpenCV, PIL
- **Model Training:** scikit-learn (for `train_test_split`), Pickle (for history)
- **Frontend/UI:** Streamlit

---

## 📂 Project Structure

```
RIFD-Net/
│── Dataset/                # Training datasets (ignored in Git)
│── model/                  # Saved weights & npy files
│   ├── rifd.keras          # Trained weights
│   ├── rifd.pckl           # Training history
│   ├── X.npy, Y.npy        # Training input/label arrays
│── model1/                 # Additional model variants
│── testImages/             # Sample images for testing
│   ├── test.png
│── streamlit_app.py        # Streamlit UI for inference
│── train.py                # Training script
│── rifd.py                 # Model utilities
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── .gitignore              # Ignore datasets/weights
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/krishnatejaswi2005/Robust-Image-Forgery-Detection-Using-Deep-Learning.git
cd Robust-Image-Forgery-Detection-Using-Deep-Learning
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset and pretrained weights

- Place your **training data** (`X.npy`, `Y.npy`) inside `model/`
- (Optional) Download pretrained weights `rifd.keras` (if available) and place in `model/`

### 5. Run training (if starting from scratch)

```bash
python train.py
```

This will train the model and save weights to `model/rifd.keras`.

### 6. Launch the Streamlit web app

```bash
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501/`

---

## 🎯 Usage

1. Open the Streamlit app.
2. Upload an image (`.png`, `.jpg`, `.jpeg`).
3. The app will:

   - Show the **original image**.
   - Display the **predicted forgery mask** side by side.

---

## 🧩 Future Improvements

- Integration with more advanced forgery detection models (e.g., CNN-LSTM, Vision Transformers).
- Support for larger input sizes (beyond 256×256).
- Benchmark evaluation on public forgery datasets.

---
