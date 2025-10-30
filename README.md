# MNIST Handwritten Digit Classification with a CNN

This project is a complete, end-to-end implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras.



## 1. Problem Statement

The goal of this project is to build and train a deep learning model that can accurately classify 28x28 grayscale images of handwritten digits (from 0 to 9). This is a classic multi-class classification problem in the field of computer vision.

## 2. Dataset Used

The model is trained on the **MNIST dataset**, which is a widely used benchmark in machine learning.
* **Training Set:** 60,000 images and their labels.
* **Testing Set:** 10,000 images and their labels.
* **Image Details:** 28x28 pixels, grayscale (1 channel).

The data was preprocessed as follows:
* Pixel values were normalized from the `[0, 255]` range to `[0.0, 1.0]`.
* Image data was reshaped to include a channel dimension: `(28, 28) -> (28, 28, 1)`.
* Labels were one-hot encoded (e.g., `7` -> `[0,0,0,0,0,0,0,1,0,0]`).

## 3. Model Architecture

A sequential CNN model was built with the following architecture:

1.  **Conv2D:** 32 filters, (3, 3) kernel, `relu` activation
2.  **MaxPooling2D:** (2, 2) pool size
3.  **Conv2D:** 64 filters, (3, 3) kernel, `relu` activation
4.  **MaxPooling2D:** (2, 2) pool size
5.  **Flatten:** Converts 2D feature maps to a 1D vector
6.  **Dense:** 64 units, `relu` activation
7.  **Dropout:** 0.5 (for regularization)
8.  **Dense (Output):** 10 units, `softmax` activation

**Model Summary:**

## 4. Instructions for Running the Code

1.  **Clone the repository:**
    ```bash
    git clone [URL_OF_YOUR_REPO]
    cd [REPO_NAME]
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the code:**
    * If you saved as a Python file (`.py`):
        ```bash
        python mnist_cnn.py
        ```
    * If you saved as a Jupyter Notebook (`.ipynb`):
        ```bash
        jupyter notebook mnist_cnn.ipynb
        ```

## 5. Evaluation Metrics and Results

The model was trained for 10 epochs and evaluated on the 10,000-image test set.

* **Test Loss:** `[Enter your test loss, e.g., 0.0285]`
* **Test Accuracy:** `[Enter your test accuracy, e.g., 99.15%]`
