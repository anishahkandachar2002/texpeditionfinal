import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
import easyocr
import requests
from PIL import Image
import io
import hashlib
from datetime import datetime
import pytz

# Set page config
st.set_page_config(
    page_title="Secure Email Spam Detector",
    page_icon="🔒📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GDPR and security compliance functions
def contains_pii(text):
    """Check for Personally Identifiable Information using regex patterns"""
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }
    
    detected_pii = {}
    for pii_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_pii[pii_type] = len(matches)
    
    return detected_pii if detected_pii else None

def anonymize_text(text):
    """Anonymize detected PII in text"""
    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    # Phone
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b', '[PHONE_REDACTED]', text)
    # Credit Card
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CC_REDACTED]', text)
    # IP Address
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', text)
    
    return text

def hash_user_data(data):
    """Create a SHA-256 hash of user data for logging"""
    return hashlib.sha256(data.encode()).hexdigest()

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv", encoding='ISO-8859-1')
    # Check the actual column names in the dataframe
    print("Columns in dataset:", df.columns.tolist())
    
    # Rename columns if needed (adjust based on actual column names)
    df.columns = ['Category', 'Message'] + list(df.columns[2:])
    
    # Drop unnecessary columns
    df = df[['Category', 'Message']].copy()
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    
    # Add text length feature
    df['Text_Length'] = df['Message'].apply(len)
    
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25, random_state=42)
    
    # Create both models for comparison
    nb_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('nb', MultinomialNB())
    ])
    
    rf_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    nb_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate models
    nb_score = nb_pipeline.score(X_test, y_test)
    rf_score = rf_pipeline.score(X_test, y_test)
    
    # Return the better performing model
    if nb_score > rf_score:
        st.session_state['model_type'] = "Naive Bayes"
        st.session_state['model_score'] = nb_score
        return nb_pipeline
    else:
        st.session_state['model_type'] = "Random Forest"
        st.session_state['model_score'] = rf_score
        return rf_pipeline

model = train_model()

def detect_spam(email_text):
    prediction = model.predict([email_text])
    prediction_proba = model.predict_proba([email_text])[0]
    return prediction[0], prediction_proba

def url_to_text(image_url, languages=['en']):
    try:
        # GDPR check - validate URL is not pointing to private resource
        if not image_url.startswith(('http://', 'https://')):
            st.error("Invalid URL format. Only HTTP/HTTPS URLs are allowed.")
            return None
            
        if any(domain in image_url for domain in ['localhost', '127.0.0.1', '192.168.', '10.']):
            st.error("Private IP addresses are not allowed for security reasons.")
            return None
            
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Check content type is image
        if 'image' not in response.headers.get('Content-Type', ''):
            st.error("URL does not point to a valid image.")
            return None
            
        image = Image.open(io.BytesIO(response.content))
        image_array = np.array(image)
        reader = easyocr.Reader(languages)
        results = reader.readtext(image_array)
        extracted_text = ""
        for (bbox, text, confidence) in results:
            extracted_text += text + " "
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def log_usage(user_input, prediction, pii_detected=None):
    """Log usage data without storing PII (for GDPR compliance)"""
    timestamp = datetime.now(pytz.utc).isoformat()
    input_hash = hash_user_data(user_input)
    log_entry = {
        'timestamp': timestamp,
        'input_hash': input_hash,
        'prediction': int(prediction),
        'pii_detected': bool(pii_detected),
        'input_length': len(user_input)
    }
    
    # In a real application, you would store this in a secure database
    # For demo purposes, we'll just keep it in session state
    if 'usage_logs' not in st.session_state:
        st.session_state['usage_logs'] = []
    st.session_state['usage_logs'].append(log_entry)
    
    return log_entry

# Main app
def main():
    st.title("🔒 Secure Email Spam Detection System")
    st.markdown("""
    This app detects whether an email is spam or ham (not spam) using machine learning.
    It includes GDPR compliance checks and security features to protect user data.
    """)
    
    # Privacy notice (GDPR requirement)
    with st.expander("Privacy Notice"):
        st.markdown("""
        **Data Protection Information:**
        - This application processes email text to determine if it's spam.
        - We automatically detect and redact Personally Identifiable Information (PII).
        - No raw email content is stored permanently.
        - Usage logs contain only anonymized, hashed data.
        - You can request deletion of your data by contacting us.
        """)
    
    # Sidebar with options
    st.sidebar.header("Options")
    input_method = st.sidebar.radio("Input Method:", ("Text Input", "Image URL"))
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Model Type: {st.session_state.get('model_type', 'Naive Bayes')}")
    st.sidebar.write(f"Accuracy: {st.session_state.get('model_score', 0):.2%}")
    
    # Data visualization options
    st.sidebar.markdown("---")
    st.sidebar.header("Data Insights")
    show_data = st.sidebar.checkbox("Show Dataset Insights")
    
    if input_method == "Text Input":
        email_text = st.text_area("Enter email text here:", height=200,
                                 help="The content will be checked for PII before processing")
        
        if st.button("Check for Spam"):
            if not email_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # GDPR compliance checks
                pii_detected = contains_pii(email_text)
                if pii_detected:
                    st.warning("⚠️ PII Detected: " + ", ".join([f"{k} ({v} instances)" for k,v in pii_detected.items()]))
                    anonymized_text = anonymize_text(email_text)
                    st.info("Anonymized version will be used for analysis:")
                    st.code(anonymized_text[:500] + ("..." if len(anonymized_text) > 500 else ""))
                    analysis_text = anonymized_text
                else:
                    analysis_text = email_text
                
                with st.spinner("Analyzing content..."):
                    result, proba = detect_spam(analysis_text)
                    log_entry = log_usage(analysis_text, result, pii_detected)
                
                if result == 0:
                    st.success(f"✅ This is Ham (Not Spam) with {proba[0]:.1%} confidence")
                else:
                    st.error(f"⚠️ This is Spam with {proba[1]:.1%} confidence")
                
                # Show detailed analysis
                with st.expander("Detailed Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Spam Probability", f"{proba[1]:.1%}")
                        st.metric("Ham Probability", f"{proba[0]:.1%}")
                    with col2:
                        st.write("Key Features:")
                        vectorizer = model.named_steps['vectorizer']
                        classifier = model.named_steps[list(model.named_steps.keys())[1]]
                        
                        if hasattr(classifier, 'feature_log_prob_'):  # Naive Bayes
                            features = vectorizer.get_feature_names_out()
                            spam_probs = classifier.feature_log_prob_[1, :]
                            top_spam_indices = spam_probs.argsort()[-10:][::-1]
                            st.write("Top spam indicators:")
                            for idx in top_spam_indices:
                                st.write(f"- {features[idx]}")
                    
                # Show word cloud
                st.subheader("Text Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Word Cloud**")
                    wordcloud = WordCloud(width=400, height=200,
                                        background_color='white',
                                        stopwords=STOPWORDS,
                                        min_font_size=10,
                                        max_words=200).generate(analysis_text)
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt)
                
                with col2:
                    st.markdown("**Text Statistics**")
                    st.write(f"Character count: {len(analysis_text)}")
                    st.write(f"Word count: {len(analysis_text.split())}")
                    st.write(f"Avg word length: {np.mean([len(word) for word in analysis_text.split()]):.1f}")
                    
                    # Check for common spam phrases
                    spam_phrases = ["click here", "limited offer", "free money", "guaranteed", "winner", "prize", "congratulations"]
                    found_phrases = [phrase for phrase in spam_phrases if phrase in analysis_text.lower()]
                    if found_phrases:
                        st.warning(f"Common spam phrases detected: {', '.join(found_phrases)}")
    
    else:  # Image URL input
        st.warning("Image processing may take longer and requires a clear image with text.")
        image_url = st.text_input("Enter image URL containing email text:", 
                                placeholder="https://example.com/email_image.png")
        languages = st.multiselect("Select languages in the image", ['en', 'es', 'fr', 'de'], default=['en'])
        
        if st.button("Extract Text and Check for Spam"):
            if not image_url:
                st.warning("Please enter an image URL.")
            else:
                with st.spinner("Processing image..."):
                    extracted_text = url_to_text(image_url, languages)
                
                if extracted_text:
                    st.subheader("Extracted Text")
                    st.text_area("", extracted_text, height=200, key="extracted_text")
                    
                    # GDPR compliance checks
                    pii_detected = contains_pii(extracted_text)
                    if pii_detected:
                        st.warning("⚠️ PII Detected: " + ", ".join([f"{k} ({v} instances)" for k,v in pii_detected.items()]))
                        anonymized_text = anonymize_text(extracted_text)
                        st.info("Anonymized version will be used for analysis:")
                        st.code(anonymized_text[:500] + ("..." if len(anonymized_text) > 500 else ""))
                        analysis_text = anonymized_text
                    else:
                        analysis_text = extracted_text
                    
                    with st.spinner("Analyzing content..."):
                        result, proba = detect_spam(analysis_text)
                        log_entry = log_usage(analysis_text, result, pii_detected)
                    
                    if result == 0:
                        st.success(f"✅ This is Ham (Not Spam) with {proba[0]:.1%} confidence")
                    else:
                        st.error(f"⚠️ This is Spam with {proba[1]:.1%} confidence")
                    
                    # Show word cloud
                    st.subheader("Text Analysis")
                    wordcloud = WordCloud(width=800, height=400,
                                        background_color='white',
                                        stopwords=STOPWORDS,
                                        min_font_size=10,
                                        max_words=1000,
                                        colormap='gist_heat_r').generate(analysis_text)
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt)
    
    # Data visualization section
    if show_data:
        st.subheader("Dataset Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spam vs Ham Distribution**")
            spread = df['Category'].value_counts()
            fig, ax = plt.subplots()
            spread.plot(kind='pie', autopct='%1.2f%%', cmap='Set1', ax=ax)
            ax.set_title('Distribution of Spam vs Ham')
            st.pyplot(fig)
            
        with col2:
            st.markdown("**Text Length Distribution**")
            fig, ax = plt.subplots()
            sns.boxplot(x='Category', y='Text_Length', data=df, ax=ax)
            ax.set_yscale('log')
            ax.set_title('Message Length by Category')
            st.pyplot(fig)
        
        st.markdown("**Spam Messages Word Cloud**")
        df_spam = df[df['Category']=='spam'].copy()
        comment_words = ' '.join(df_spam.Message.astype(str))
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white',
                            stopwords=STOPWORDS,
                            min_font_size=10,
                            max_words=1000,
                            colormap='gist_heat_r').generate(comment_words)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)
        
        # Show sample spam and ham messages
        st.markdown("**Sample Messages**")
        tab1, tab2 = st.tabs(["Spam Examples", "Ham Examples"])
        with tab1:
            st.write(df[df['Category']=='spam'].head(5)[['Message']])
        with tab2:
            st.write(df[df['Category']=='ham'].head(5)[['Message']])
    
    # Usage logs (GDPR compliance)
    if st.sidebar.checkbox("Show Usage Logs (Admin)"):
        st.subheader("Usage Logs (Anonymized)")
        if 'usage_logs' in st.session_state and st.session_state['usage_logs']:
            logs_df = pd.DataFrame(st.session_state['usage_logs'])
            st.dataframe(logs_df)
            
            # Statistics
            st.write(f"Total analyses: {len(logs_df)}")
            st.write(f"Spam detected: {logs_df['prediction'].sum()} times")
            st.write(f"PII detected: {logs_df['pii_detected'].sum()} times")
        else:
            st.info("No usage logs available yet.")

if __name__ == "__main__":
    main()