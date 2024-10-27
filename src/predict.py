import os
import joblib

def load_model_and_predict(email):
    if not email.strip():
        return "Invalid input: Email content cannot be empty."

    # Ensure the model files exist
    if not os.path.exists('models/spam_detector_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
        return "Error: Model or vectorizer files are missing."

    # Load the trained model and vectorizer
    model = joblib.load('models/spam_detector_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')

    # Transform the email input and predict
    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)

    return 'Spam' if prediction[0] == 1 else 'Not Spam'
