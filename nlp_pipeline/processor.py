import os
import importlib
import nltk
import pkg_resources

class NLPProcessor:
    def __init__(self, stem=False, lemmatize=False, vectorize=None, backend="nltk"):
        self.stem = stem
        self.lemmatize = lemmatize
        self.vectorize = vectorize
        self.backend = backend.lower()
        
        self.ensure_dependencies()
        
        # Dynamically import libraries
        self.nltk = importlib.import_module("nltk")
        self.sklearn = importlib.import_module("sklearn.feature_extraction.text")
        self.spacy = importlib.import_module("spacy") if self.backend == "spacy" else None
        
        # Load NLP components dynamically
        self.stemmer = self.get_stemmer() if self.stem else None
        self.lemmatizer = self.get_lemmatizer() if self.lemmatize else None
        self.vectorizer_model = self.get_vectorizer() if self.vectorize else None
        
    def ensure_dependencies(self):
        required_packages = ["nltk", "spacy", "scikit-learn"]
        missing_packages = [pkg for pkg in required_packages if not self.is_installed(pkg)]
        
        if missing_packages:
            os.system(f"pip install {' '.join(missing_packages)}")
        
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        os.system("python -m spacy download en_core_web_sm")
    
    def is_installed(self, package):
        try:
            pkg_resources.get_distribution(package)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    
    def get_stemmer(self):
        try:
            return self.nltk.stem.PorterStemmer()
        except AttributeError:
            return None
    
    def get_lemmatizer(self):
        if self.backend == "nltk":
            try:
                return self.nltk.stem.WordNetLemmatizer()
            except AttributeError:
                return None
        elif self.backend == "spacy":
            try:
                return self.spacy.load("en_core_web_sm")
            except OSError:
                return None
    
    def get_vectorizer(self):
        try:
            if self.vectorize == "tfidf":
                return self.sklearn.TfidfVectorizer()
            elif self.vectorize == "count":
                return self.sklearn.CountVectorizer()
        except AttributeError:
            return None
    
    def process(self, text):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        words = text.split()
        
        if self.stem and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        
        if self.lemmatize:
            if self.backend == "nltk" and self.lemmatizer:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            elif self.backend == "spacy" and self.lemmatizer:
                doc = self.lemmatizer(" ".join(words))
                words = [token.lemma_ for token in doc]
        
        processed_text = " ".join(words)
        
        if self.vectorizer_model:
            vectorized_output = self.vectorizer_model.fit_transform([processed_text])
            return vectorized_output.toarray()
        
        return processed_text
    
    @staticmethod
    def supported_vectorizers():
        return ["tfidf", "count"]

# Example Usage:
if __name__ == "__main__":
    processor = NLPProcessor(stem=True, lemmatize=True, vectorize="tfidf", backend="spacy")
    output = processor.process("running jumped swimming")
    print(output)
