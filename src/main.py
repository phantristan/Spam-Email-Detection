from predict import load_model_and_predict

def main():
    print("Welcome to the Spam Email Detection CLI!")
    while True:
        email = input("\nEnter the email content (or type 'exit' to quit): ")
        if email.lower() == 'exit':
            break
        print("Prediction:", load_model_and_predict(email))

if __name__ == '__main__':
    main()