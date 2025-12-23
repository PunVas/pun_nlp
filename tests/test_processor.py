import unittest
import numpy as np
from pun_nlp import NLPProcessor

class TestNLPProcessor(unittest.TestCase):
    def test_initialization(self):
        """Test basic initialization defaults."""
        processor = NLPProcessor()
        self.assertFalse(processor.tokenize)
        self.assertIsNone(processor.vectorizer)

    def test_lazy_loading(self):
        """Test that resources are not loaded immediately."""
        processor = NLPProcessor(tokenize=True)
        # Assuming resources aren't pre-loaded on system or checking internal flag
        # Ideally we'd mock nltk, but for integration:
        self.assertFalse(processor._resources_downloaded)
        processor.process("Test")
        self.assertTrue(processor._resources_downloaded)

    def test_processing_basic(self):
        """Test simple processing."""
        processor = NLPProcessor(tokenize=True, normalize=True)
        result = processor.process("Hello World!")
        # normalize -> hello world -> tokens: ['hello', 'world']
        # actually normalize removes punctuation: "hello world" -> ['hello', 'world']
        self.assertEqual(result, ['hello', 'world'])

    def test_stopwords_stemming(self):
        """Test stopwords removal and stemming."""
        processor = NLPProcessor(tokenize=True, remove_stopwords=True, stem=True)
        # "running" -> "run", "is" -> stopword
        result = processor.process("running is fun")
        # 'is' removed. 'running' -> 'run'. 'fun' -> 'fun'
        self.assertIn('run', result)
        self.assertNotIn('is', result)

    def test_vectorization_error_conflicts(self):
        """Test that conflicting options raise ValueError."""
        with self.assertRaises(ValueError):
            NLPProcessor(vectorize="tfidf", pos_tagging=True)

    def test_vectorization_tfid(self):
        """Test TF-IDF vectorization."""
        processor = NLPProcessor(vectorize="tfidf", tokenize=True) 
        texts = ["hello world", "hello python"]
        processor.fit_vectorizer(texts)
        vectors = processor.transform_texts(texts)
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape[0], 2) # 2 documents

    def test_2d_input(self):
        """Test 2D list input."""
        processor = NLPProcessor(tokenize=True)
        input_data = [["hello"], ["world"]]
        result = processor.process(input_data)
        self.assertEqual(result, [[['hello']], [['world']]])

if __name__ == '__main__':
    unittest.main()
