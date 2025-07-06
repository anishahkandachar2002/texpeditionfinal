import pickle

# Load the trained model
def load_model():
    with open('spam_detector_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Predict if email is spam or not
def predict_spam(email_text, model):
    prediction = model.predict([email_text])
    probability = model.predict_proba([email_text])
    
    result = "SPAM" if prediction[0] == 1 else "HAM (Not Spam)"
    confidence = max(probability[0]) * 100
    
    return result, confidence

# Main function
def main():
    # Load the model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully!")
    
    while True:
        print("\n" + "="*50)
        print("Email Spam Detector")
        print("="*50)
        
        email_text = input("\nEnter email text (or 'quit' to exit): ")
        
        if email_text.lower() == 'quit':
            print("Goodbye!")
            break
        
        if email_text.strip() == "":
            print("Please enter some text to analyze.")
            continue
        
        # Ask for confirmation before analyzing
        confirm = input("\nPress 'y' to analyze this email: ").lower()
        
        if confirm == 'y':
            # Make prediction
            result, confidence = predict_spam(email_text, model)
            
            print(f"\nPrediction: {result}")
            print(f"Confidence: {confidence:.2f}%")
            
            if result == "SPAM":
                print("⚠️  This email appears to be spam!")
            else:
                print("✅ This email appears to be legitimate.")
        else:
            print("Analysis cancelled.")

if __name__ == "__main__":
    main()