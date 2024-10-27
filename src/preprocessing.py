import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path='data/email.csv'):
    # Load dataset
    data = pd.read_csv(file_path)

    # Select and rename columns
    data.columns = data.columns.str.lower() # Normalize column names to lowercase
    data = data[['category', 'message']]
    data.columns = ['label', 'message'] # Rename columns

    # Map labels to binary values (ham = 0, spam = 1)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Check for any unmapped or NaN values in 'label'
    if data['label'].isnull().any():
        print("Found unmapped labels. Dropping them.")
        print(data[data['label'].isnull()])  # Print problematic rows
        data = data.dropna(subset=['label'])  # Drop rows with NaN in 'label'

    # Split dataset into training and test sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.3, random_state=42
    )

    # Convert text data into numerical features using TF_IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
