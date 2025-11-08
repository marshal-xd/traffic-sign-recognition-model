# 🚦 Traffic Sign Recognition Model (TSRIM)

The **Traffic Sign Recognition Model (TSRIM)** is a deep learning project utilizing a **Convolutional Neural Network (CNN)** to automatically classify traffic signs. This model is designed to recognize and distinguish between **43 distinct classes** of German traffic signs, a fundamental task in the development of Advanced Driver-Assistance Systems (ADAS) and autonomous vehicles.

-----

## 🚀 Key Features

  * **Deep Learning Architecture:** Implements a Sequential CNN using **Keras** and **TensorFlow** for robust feature extraction.
  * **High Accuracy:** The model achieves a **validation accuracy of 98.98%** and an external **test accuracy of 95.52%**.
  * **Complete Pipeline:** The included Jupyter Notebook (`TSRIMCODE.ipynb`) covers data loading, preprocessing, model training, evaluation, and visualization.
  * **Modular Design:** The trained model is saved separately (`TSR.keras`) for easy deployment and inference.

-----

## 📚 Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which was sourced from **Kaggle**.

The GTSRB dataset contains a large number of real-world traffic sign images, essential for training a reliable recognition system.

  * **Total Images:** The combined training and testing set contains **43,905 images**.
  * **Input Standardization:** All images are preprocessed and resized to a consistent **30x30x3 (RGB)** format before being fed into the network.

-----

## 🧠 Model Details

The model is a deep CNN designed for multi-class classification:

| Component | Description |
| :--- | :--- |
| **Model Type** | Sequential CNN |
| **Input Shape** | 30 x 30 x 3 (RGB) |
| **Layers** | Multiple `Conv2D`, `MaxPool2D`, and **`Dropout`** layers (to prevent overfitting) leading to `Flatten` and `Dense` layers. |
| **Output** | 43 classes with `softmax` activation. |
| **Optimizer** | **Adam** |
| **Loss Function** | Categorical Crossentropy |
| **Training** | Trained for **20 epochs**. |

### **Important Note on Class Labels**

The model output is a class index (0 to 42). To interpret the results, you must use the traffic sign class mapping defined within the notebook (`TSRIMCODE.ipynb`) to match the index to the actual traffic sign (e.g., speed limit, stop, yield).

-----

## 🛠️ Setup and Usage

To run the project locally, follow these steps:

### **1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <project-folder-name>
```

### **2. Acquire the Dataset**

Download the **GTSRB dataset** from Kaggle and ensure the folder structure (containing `Train` and `Test` directories) is correctly placed relative to the notebook, or update the `project_path` variable within `TSRIMCODE.ipynb`.

### **3. Install Dependencies**

This project requires standard deep learning and image processing libraries. It is recommended to use a virtual environment.

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python Pillow
```

### **4. Run the Notebook**

Open and run the `TSRIMCODE.ipynb` in a Jupyter environment to execute the full pipeline: data preparation, model training, and final evaluation. The trained model will be saved as `TSR.keras`.

### **5. Making Predictions**

You can load the saved model directly for inference on new images:

```python
from keras.models import load_model

# Load the trained model
model = load_model('./training/TSR.keras') 

# Preprocess your new image (resize to 30x30, normalize, reshape) and then predict
# predictions = model.predict(preprocessed_image)
```
