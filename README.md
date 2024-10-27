# **Spam Email Detection Using Machine Learning**

## **Project Overview**
This project demonstrates the development of a **machine learning-based spam email detection system** using the **Naive Bayes algorithm**. The goal is to classify emails as **spam or non-spam** using text-based features from email content, subject lines, and sender details. This solution automates email filtering to improve productivity and enhance security.

---

## **Features**
- **Naive Bayes Classifier:** A supervised learning algorithm tailored for text classification tasks like spam filtering.
- **TF-IDF Vectorization:** Converts email content into numerical features for training the model.
- **Interactive CLI Interface:** Users can input email content and receive real-time predictions.
- **Visualization Tools:** Includes bar charts, word clouds, and confusion matrices to explore spam patterns and model performance.
- **Deployable via Google Colab:** Run the notebook online with no local setup required.

---

## **Project Structure**
/spam-email-detection
│
├── spam_detection_notebook.ipynb   # Jupyter notebook with code, visualizations, and instructions
│
├── src/                            # Source code directory
│   ├── main.py                     # CLI interface for running predictions
│   ├── predict.py                  # Script for loading the model and making predictions
│   ├── preprocessing.py            # Data preprocessing and cleaning logic
│   ├── train_model.py              # Model training script
│   ├── data/                       # Directory containing datasets
│   │   └── email.csv               # Labeled spam and non-spam email dataset
│   └── models/                     # Directory for trained models and vectorizer
│       ├── spam_detector_model.pkl # Trained Naive Bayes model
│       └── vectorizer.pkl          # Trained TF-IDF vectorizer


---

## **How to Run the Project**

### **Option 1: Run on Google Colab**
1. **Open the notebook** in [Google Colab](https://colab.research.google.com/).
2. **Upload `email.csv`** to the session or load it from GitHub (see code below).
3. **Install required dependencies** by running:
   ```python
   !pip install pandas scikit-learn matplotlib seaborn wordcloud joblib
   
### **Option 2: Run Locally**
1. **Clone this respository:**
   ```python
   git clone https://github.com/phantristan/Spam-Email-Detection.git 
2. **Install required dependencies** by running:
   ```python
   !pip install pandas scikit-learn matplotlib seaborn wordcloud joblib
3. **Run the CLI interface:**
   ```python
   python src/main.py
4. **Enter sample email content when prompted.**

---

## **Dataset**
The dataset used in this project is a public dataset containing labeled spam and non-spam emails.

**To load the dataset directly from GitHub**
```python
import pandas as pd

url = 'https://raw.githubusercontent.com/phantristan/Spam-Email-Drection/main/src/data/email.csv'
data = pd.read_csv(url) 
```
**To download the dataset directly from its origin (kaggle.com)**
https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification


