import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
import easyocr
import requests
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25)
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X_train, y_train)
    return clf

model = train_model()

def detect_spam(email_text):
    prediction = model.predict([email_text])
    return prediction[0]

def url_to_text(image_url, languages=['en']):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image_array = np.array(image)
        reader = easyocr.Reader(languages)
        results = reader.readtext(image_array)
        extracted_text = ""
        for (bbox, text, confidence) in results:
            extracted_text += text + " "
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Main app
def main():
    st.title("📧 Email Spam Detection System")
    st.markdown("""
    This app detects whether an email is spam or ham (not spam) using a Naive Bayes classifier.
    You can either type/paste email text or upload an image containing text.
    """)
    
    # Sidebar with options
    st.sidebar.header("Options")
    input_method = st.sidebar.radio("Input Method:", ("Text Input", "Image URL"))
    
    if input_method == "Text Input":
        email_text = st.text_area("Enter email text here:", height=200)
        if st.button("Check for Spam"):
            if email_text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                result = detect_spam(email_text)
                if result == 0:
                    st.success("✅ This is a Ham Email (Not Spam)!")
                else:
                    st.error("⚠️ This is a Spam Email!")
                    
                # Show word cloud for spam
                if result == 1:
                    st.subheader("Spam Word Cloud")
                    wordcloud = WordCloud(width=800, height=400,
                                        background_color='white',
                                        stopwords=STOPWORDS,
                                        min_font_size=10,
                                        max_words=1000,
                                        colormap='gist_heat_r').generate(email_text)
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt)
    
    else:  # Image URL input
        image_url = st.text_input("Enter image URL containing email text:")
        languages = st.multiselect("Select languages in the image", ['en', 'es', 'fr', 'de'], default=['en'])
        
        if st.button("Extract Text and Check for Spam"):
            if not image_url:
                st.warning("Please enter an image URL.")
            else:
                with st.spinner("Processing image..."):
                    extracted_text = url_to_text(image_url, languages)
                    
                if extracted_text:
                    st.subheader("Extracted Text")
                    st.text_area("", extracted_text, height=200)
                    
                    result = detect_spam(extracted_text)
                    if result == 0:
                        st.success("✅ This is a Ham Email (Not Spam)!")
                    else:
                        st.error("⚠️ This is a Spam Email!")
                        
                    # Show word cloud for spam
                    if result == 1:
                        st.subheader("Spam Word Cloud")
                        wordcloud = WordCloud(width=800, height=400,
                                            background_color='white',
                                            stopwords=STOPWORDS,
                                            min_font_size=10,
                                            max_words=1000,
                                            colormap='gist_heat_r').generate(extracted_text)
                        plt.figure(figsize=(10,5))
                        plt.imshow(wordcloud)
                        plt.axis("off")
                        st.pyplot(plt)
    
    # Data visualization section
    st.sidebar.header("Data Insights")
    if st.sidebar.checkbox("Show Dataset Info"):
        st.subheader("Dataset Information")
        st.write(f"Total emails: {len(df)}")
        st.write(df.head())
        
        st.subheader("Spam vs Ham Distribution")
        spread = df['Category'].value_counts()
        fig, ax = plt.subplots()
        spread.plot(kind='pie', autopct='%1.2f%%', cmap='Set1', ax=ax)
        ax.set_title('Distribution of Spam vs Ham')
        st.pyplot(fig)
        
        st.subheader("Spam Messages Word Cloud")
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

if __name__ == "__main__":
    main()