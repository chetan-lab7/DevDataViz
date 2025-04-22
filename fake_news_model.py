import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
import logging
import pickle
import os
import re
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set up NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK data
def download_nltk_data():
    max_retries = 3
    for i in range(max_retries):
        try:
            nltk.download('punkt', download_dir=nltk_data_dir)
            nltk.download('stopwords', download_dir=nltk_data_dir)
            logging.info("NLTK data downloaded successfully")
            return True
        except Exception as e:
            if i < max_retries - 1:
                logging.warning(f"NLTK download attempt {i+1} failed: {str(e)}. Retrying...")
                time.sleep(1)  # Wait before retrying
            else:
                logging.error(f"Failed to download NLTK data after {max_retries} attempts: {str(e)}")
                return False

# Download NLTK data
download_nltk_data()

# Create our own tokenizer function instead of using word_tokenize
def simple_word_tokenize(text):
    """A simple tokenizer that splits text on whitespace and punctuation"""
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace
    return text.split()

class FakeNewsDetector:
    def __init__(self):
        """Initialize the fake news detector with a pre-trained model."""
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Train a simple model with some fake and real news examples
        self._train_model()
    
    def _preprocess_text(self, text):
        """Preprocess the text by removing punctuation, stopwords, and stemming."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Tokenize using our custom tokenizer
        tokens = simple_word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def _train_model(self):
        """Train a simple model with some examples."""
        # Define some example fake and real news
        # Note: This is a very simplified training set for demonstration purposes
        fake_news = [
            "BREAKING: Aliens confirmed to be living among us, government has been hiding this for decades!",
            "SHOCKING: Scientists discover that drinking water causes cancer!",
            "EXCLUSIVE: Celebrity secretly a robot controlled by government agents!",
            "URGENT: 5G networks are spreading deadly diseases!",
            "Secret document reveals world leaders are actually lizard people in disguise!",
            "Doctor discovers miracle cure for all diseases, big pharma tries to silence him!",
            "Government implanting microchips in vaccines to track citizens!",
            "Study shows chocolate is more addictive than cocaine, sugar industry covers it up!",
            "Scientist proves that the Earth is actually flat, NASA has been lying!"
        ]
        
        real_news = [
            "Study shows moderate exercise can improve heart health according to medical researchers.",
            "New species of frog discovered in Amazon rainforest by biology expedition team.",
            "Local council approves funding for road infrastructure improvements starting next month.",
            "Tech company releases quarterly financial report showing 3% growth in revenue.",
            "Weather forecast predicts rain and cooler temperatures for the upcoming weekend.",
            "University researchers publish findings on climate change effects in peer-reviewed journal.",
            "Stock market closes with mixed results after Federal Reserve announcement on interest rates.",
            "New restaurant opens downtown featuring locally sourced ingredients and seasonal menu.",
            "International diplomatic talks continue as nations work toward peace agreement."
        ]
        
        # Combine and label the data
        X = fake_news + real_news
        y = [1] * len(fake_news) + [0] * len(real_news)  # 1 for fake, 0 for real
        
        # Preprocess the data
        X_processed = [self._preprocess_text(text) for text in X]
        
        # Train the vectorizer and model
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        self.model.fit(X_vectorized, y)
        
        logging.info("Fake news detection model trained successfully")
    
    def predict(self, news_content):
        """
        Predict whether the given news content is fake or real.
        
        Args:
            news_content (str): The news content to analyze
            
        Returns:
            tuple: (prediction, confidence)
                - prediction (str): 'FAKE' or 'REAL'
                - confidence (float): Confidence score (0-100%)
        """
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(news_content)
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([processed_text])
            
            # Get prediction and probability
            prediction = self.model.predict(text_vectorized)[0]
            prob = self.model.predict_proba(text_vectorized)[0]
            
            # Determine confidence
            confidence = prob[1] if prediction == 1 else prob[0]
            confidence = round(confidence * 100, 2)
            
            # Return prediction and confidence
            result = 'FAKE' if prediction == 1 else 'REAL'
            return result, confidence
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
