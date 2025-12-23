import re
import string
import logging
from typing import List, Optional, Union, Any, Tuple, Set

import nltk
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    A robust, NLP pipeline for text preprocessing and vectorization.
    
    Attributes:
        stem (bool): Enable stemming using PorterStemmer.
        lemmatize (bool): Enable lemmatization using WordNetLemmatizer.
        vectorize (Optional[str]): Vectorization method ('tfidf', 'count') or None.
        tokenize (bool): Enable tokenization.
        remove_stopwords (bool): Remove English stopwords.
        pos_tagging (bool): Enable Part-of-Speech tagging.
        ner (bool): Enable Named Entity Recognition (requires spaCy).
        normalize (bool): Lowercase and remove punctuation.
        backend (str): backend library for specific tasks ('nltk', 'spacy').
    """
    
    def __init__(
        self, 
        stem: bool = False, 
        lemmatize: bool = False, 
        vectorize: Optional[str] = None, 
        tokenize: bool = False, 
        remove_stopwords: bool = False,
        pos_tagging: bool = False, 
        ner: bool = False, 
        normalize: bool = False, 
        backend: str = "nltk"
    ):
        self.stem = stem
        self.lemmatize = lemmatize
        self.vectorize = vectorize
        self.tokenize = tokenize
        self.remove_stopwords = remove_stopwords
        self.pos_tagging = pos_tagging
        self.ner = ner
        self.normalize = normalize
        self.backend = backend.lower()
        
        if self.vectorize and (self.pos_tagging or self.ner):
            raise ValueError("Vectorization cannot be combined with POS tagging or NER.")

        self._resources_downloaded = False
        self._spacy_loaded = False
        
        self.stemmer: Optional[PorterStemmer] = None
        self.lemmatizer: Optional[WordNetLemmatizer] = None
        self.stop_words: Set[str] = set()
        self.spacy_model: Any = None
        self.vectorizer: Optional[Union[TfidfVectorizer, CountVectorizer]] = None

        if self.vectorize == "tfidf":
            self.vectorizer = TfidfVectorizer()
        elif self.vectorize == "count":
            self.vectorizer = CountVectorizer()

    def _ensure_resources(self) -> None:
        """
        Lazily downloads NLTK resources. 
        Ensures robust path handling by appending a local ./nltk_data directory
        if standard system paths are not working/writable.
        """
        if self._resources_downloaded:
            return

        try:
            import ssl
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        import os
        local_nltk_path = os.path.join(os.getcwd(), 'nltk_data')
        
        if local_nltk_path not in nltk.data.path:
            nltk.data.path.append(local_nltk_path)

        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]

        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info(f"Downloading NLTK resource: {name}")
                try:
                    nltk.download(name, download_dir=local_nltk_path, quiet=True)
                except Exception as e:
                    logger.warning(f"Download to {local_nltk_path} failed ({e}). Trying default location...")
                    nltk.download(name, quiet=True)
                
                try:
                    nltk.data.find(path)
                except LookupError:
                    zip_path = os.path.join(local_nltk_path, path + ".zip")
                    if os.path.exists(zip_path):
                        try:
                            import zipfile
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                extract_root = os.path.dirname(os.path.join(local_nltk_path, path))
                                zip_ref.extractall(extract_root)
                        except Exception as e:
                            logger.error(f"Manual extraction failed: {e}")
                    
                    try:
                        nltk.data.find(path)
                    except LookupError:
                        raise RuntimeError(f"Failed to download/find NLTK resource '{name}'.")

        self._resources_downloaded = True
            
        if self.stem:
            self.stemmer = PorterStemmer()
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

    def _ensure_spacy(self) -> None:
        """Lazily loads spaCy model using auto-download."""
        if not self._spacy_loaded and (self.backend == "spacy" or self.ner):
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
                self._spacy_loaded = True
            except OSError:
                logger.info("Downloading spaCy model 'en_core_web_sm'...")
                try:
                    from spacy.cli import download
                    download("en_core_web_sm")
                    self.spacy_model = spacy.load("en_core_web_sm")
                    self._spacy_loaded = True
                except Exception as e:
                    logger.error(f"Failed to auto-download spaCy model: {e}")
                    raise

    def _normalize_text(self, text: str) -> str:
        """Lowercases and removes punctuation."""
        text = text.lower()
        # More efficient punctuation removal
        return text.translate(str.maketrans('', '', string.punctuation))

    def _process_text(self, text: str) -> Union[List[str], List[Tuple[str, str]], List[Any]]:
        """Processes a single text input string."""
        self._ensure_resources()
        
        # NER priority (uses raw text -> preserves capitalization for model accuracy)
        if self.ner:
            self._ensure_spacy()
            doc = self.spacy_model(text)
            return [(ent.text, ent.label_) for ent in doc.ents]

        if self.normalize:
            text = self._normalize_text(text)

        tokens: List[str] = word_tokenize(text) if self.tokenize or self.remove_stopwords or self.stem or self.lemmatize or self.pos_tagging else [text] # type: ignore 
        
        # Branch 1: POS Tagging (Needs original words/context for accuracy)
        if self.pos_tagging:
            # Tag the FULL sentence first to preserve context
            tagged_tokens = nltk.pos_tag(tokens)
            
            # Remove stopwords AFTER tagging, but BEFORE stemming/lemma
            if self.remove_stopwords:
                tagged_tokens = [(w, t) for w, t in tagged_tokens if w.lower() not in self.stop_words]

            # Then Stem/Lemma the word part if requested
            if self.stem and self.stemmer:
                tagged_tokens = [(self.stemmer.stem(w), t) for w, t in tagged_tokens] # type: ignore
            
            if self.lemmatize and self.lemmatizer:
                tagged_tokens = [(self.lemmatizer.lemmatize(w), t) for w, t in tagged_tokens]
                
            return tagged_tokens

        # Branch 2: Standard Token Processing
        if self.remove_stopwords:
            tokens = [word for word in tokens if word.lower() not in self.stop_words]
            
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens] # type: ignore
        
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return tokens

    def process(self, texts: Union[str, List[str], List[List[str]]]) -> Any:
        """
        Main processing method.
        
        Args:
            texts: Single string, list of strings, or list of lists of strings.
            
        Returns:
            Processed text(s) in the corresponding structure.
        """
        if isinstance(texts, str):
            return self._process_text(texts)
        
        elif isinstance(texts, list):
            if not texts:
                return []
            if isinstance(texts[0], list):  # 2D array
                return [[self._process_text(t) for t in sublist] for sublist in texts] # type: ignore
            else:  # 1D list
                return [self._process_text(text) for text in texts] # type: ignore
        
        return None

    def fit_vectorizer(self, texts: List[str]) -> None:
        """Fits the vectorizer to the given texts."""
        processed_data = self.process(texts)
        
        if isinstance(processed_data, list) and len(processed_data) > 0 and isinstance(processed_data[0], list):
            processed_strings = [" ".join(tokens) for tokens in processed_data] # type: ignore
        else:
            processed_strings = processed_data # type: ignore

        self.vectorizer.fit(processed_strings)

    def transform_texts(self, texts: List[str]) -> Any:
        """Transforms texts using the fitted vectorizer."""
        if not self.vectorizer:
            return None
            
        processed_data = self.process(texts)
        if isinstance(processed_data, list) and len(processed_data) > 0 and isinstance(processed_data[0], list):
             processed_strings = [" ".join(tokens) for tokens in processed_data] # type: ignore
        else:
            processed_strings = processed_data # type: ignore
            
        try:
             return self.vectorizer.transform(processed_strings).toarray()
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            raise

    @staticmethod
    def supported_vectorizers() -> List[str]:
        return ["tfidf", "count"]



