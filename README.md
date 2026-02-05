# Skin Cancer Detection Model (ISIC Dataset)

## 📌 Project Overview
This project builds a Convolutional Neural Network (CNN) using **TensorFlow** and **Keras** to detect and classify 9 different types of skin cancer. The model is trained on the **ISIC (International Skin Imaging Collaboration)** dataset.

To address class imbalance in the original dataset, this project utilizes the `Augmentor` library to generate synthetic samples, ensuring the model trains on a balanced dataset for better generalization.

## 📊 Dataset
The dataset used is the **Skin cancer ISIC The International Skin Imaging Collaboration** dataset, sourced via Kaggle.

* **Source:** [Kaggle - Skin Cancer ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
* **Classes:** The dataset classifies skin lesions into **9 distinct categories**.
* **Data Imbalance Strategy:** The original dataset has varying numbers of images per class. We use an augmentation pipeline to oversample underrepresented classes, ensuring each class has ~1000 samples.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Image Augmentation:** Augmentor
* **Dataset Retrieval:** KaggleHub

## 🧠 Model Architecture
The model is a custom Convolutional Neural Network (CNN) designed for image classification:

1.  **Rescaling Layer:** Normalizes pixel values to the [0, 1] range.
2.  **Convolutional Blocks:** Five blocks of `Conv2D` layers (increasing filters: 32 → 512) followed by `MaxPool2D`.
3.  **Regularization:** `Dropout` layers (0.15, 0.20, 0.25) are added after the deeper layers to prevent overfitting.
4.  **Dense Layers:**
    * `Flatten` layer to convert 2D features to 1D.
    * Fully connected `Dense` layer with 1024 neurons (ReLU activation).
    * **Output Layer:** `Dense` layer with 9 neurons (Softmax activation) for multi-class classification.

**Optimizer:** Adam (`learning_rate=0.001`)  
**Loss Function:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy

## 🚀 How to Run the Project

### Option 1: Google Colab (Recommended)
Since the code uses `kagglehub` and requires GPU acceleration for training, Google Colab is the easiest way to run this.

1.  Upload the `.ipynb` file to your Google Drive.
2.  Open the file with **Google Colab**.
3.  Change the runtime type to **GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
4.  Run the cells sequentially. The code handles dataset downloading automatically.

### Option 2: Local Environment
If running locally, ensure you have a GPU set up with CUDA.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bhaveshv24/SkinCancerDetectionModel.git](https://github.com/bhaveshv24/SkinCancerDetectionModel.git)
    cd SkinCancerDetectionModel
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy pandas matplotlib seaborn augmentor kagglehub
    ```

3.  **Run the notebook:**
    Launch Jupyter Notebook or JupyterLab and open the file.
    ```bash
    jupyter notebook Tensorflow_Project_Skin_Cancer_Detection.ipynb
    ```

## 📉 Workflow Explanation
1.  **Data Loading:** The dataset is downloaded directly using the `kagglehub` library.
2.  **Visualization:** Initial distribution of classes is visualized to identify imbalances.
3.  **Augmentation Pipeline:**
    * The `Augmentor` library creates a pipeline to rotate and zoom images.
    * It samples 1000 images for *every* class to balance the distribution.
4.  **Model Training:** The CNN is trained for **25 epochs** on the balanced dataset.
5.  **Evaluation:**
    * Training vs. Validation Accuracy/Loss graphs are plotted.
    * The model predicts classes on unseen test data.

## 📈 Results
* The model tracks accuracy and loss over 25 epochs.
* Final validation accuracy and loss metrics are visualized in the notebook using Matplotlib.
* Sample predictions are displayed to verify model performance visually.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/bhaveshv24/SkinCancerDetectionModel/issues).

## 📜 License
This project is open-source and available under the MIT License.