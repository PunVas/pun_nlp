import nltk
import spacy
import string
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

class NLPProcessor:
    def __init__(self, stem=False, lemmatize=False, vectorize=None, tokenize=False, remove_stopwords=False,
                 pos_tagging=False, ner=False, normalize=False, backend="nltk"):
        self.stem = stem
        self.lemmatize = lemmatize
        self.vectorize = vectorize
        self.tokenize = tokenize
        self.remove_stopwords = remove_stopwords
        self.pos_tagging = pos_tagging
        self.ner = ner
        self.normalize = normalize
        self.backend = backend.lower()
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.spacy_model = None
        if self.backend == "spacy" or self.ner:
            self.spacy_model = spacy.load("en_core_web_sm")
        
        if self.vectorize == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif self.vectorize == "count":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = None

    def _normalize_text(self, text):
        """Lowercases and removes punctuation"""
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text
    
    def _process_text(self, text):
        """Processes a single text input"""
        if self.normalize:
            text = self._normalize_text(text)
        
        tokens = word_tokenize(text) if self.tokenize else [text]
        
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        if self.stem:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        if self.pos_tagging:
            tokens = nltk.pos_tag(tokens)
        
        if self.ner and self.spacy_model:
            doc = self.spacy_model(text)
            tokens = [(ent.text, ent.label_) for ent in doc.ents]
        
        return tokens
    
    def process(self, texts):
        """Processes a list or 2D array of texts"""
        if isinstance(texts, str):
            return self._process_text(texts)
        
        elif isinstance(texts, list):
            if all(isinstance(sublist, list) for sublist in texts):  # 2D array
                return [[self._process_text(text) for text in sublist] for sublist in texts]
            else:  # 1D list
                return [self._process_text(text) for text in texts]
        
        return None
    
    def fit_vectorizer(self, texts):
        """Fits the vectorizer to the given texts"""
        if self.vectorizer:
            texts = [" ".join(self._process_text(text)) for text in texts]
            self.vectorizer.fit(texts)
    
    def transform_texts(self, texts):
        """Transforms texts using the fitted vectorizer"""
        if self.vectorizer:
            texts = [" ".join(self._process_text(text)) for text in texts]
            return self.vectorizer.transform(texts).toarray()
    
    @staticmethod
    def supported_vectorizers():
        return ["tfidf", "count"]


