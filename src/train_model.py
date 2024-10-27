import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from preprocessing import load_and_preprocess_data

def train_and_save_model():
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model's performance
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model and vectorizer
    joblib.dump(model, 'models/spam_detector_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if __name__ == '__main__':
    train_and_save_model()