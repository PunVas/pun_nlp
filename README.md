# pun_nlp
[![PyPI Downloads](https://static.pepy.tech/badge/pun-nlp)](https://pepy.tech/projects/pun-nlp)

## Overview
pun_nlp is a robust NLP abstraction layer designed to simplify text processing and vectorization. It handles dependency management, resource downloading, and text preprocessing automatically, so you don't have to write boilerplate code.

It solves common issues with NLTK downloads and path errors by implementing a robust, lazy-loading resource manager that works in restricted environments like Kaggle and corporate servers.

## Features
-   **Robust Resource Management**: Automatically handles NLTK/Spacy downloads and SSL errors.
-   **Lazy Loading**: Resources are only loaded into memory when needed.
-   **Type Safety**: Prevents invalid combinations of operations (like vectorizing POS tuples).
-   **Unified API**: Process single strings, lists, or 2D arrays of text with one method.
-   **Seamless Vectorization**: Integrates directly with Scikit-Learn's TF-IDF and Count vectorizers.

## Installation
```bash
pip install pun_nlp
```

## Usage

### Basic Pipeline
```python
from pun_nlp import NLPProcessor

# Initialize with desired flags
p = NLPProcessor(
    tokenize=True, 
    stem=True, 
    remove_stopwords=True,
    normalize=True
)

text = "The QUICK brown foxes are running fast!"

# Automatically handles downloads and processing
print(p.process(text))
# Output: ['quick', 'brown', 'fox', 'run', 'fast']
```

### NER & POS Tagging
```python
# NER (Case sensitive checking happens before normalization)
p_ner = NLPProcessor(ner=True)
print(p_ner.process("Apple Inc. is hiring in California."))
# Output: [('Apple Inc.', 'ORG'), ('California', 'GPE')]

# POS Tagging (Tags tokens correctly before stemming)
p_pos = NLPProcessor(pos_tagging=True, stem=True)
print(p_pos.process("The boys are likely running."))
# Output: [('the', 'DT'), ('boy', 'NNS'), ('are', 'VBP'), ('like', 'RB'), ('run', 'VBG')]
```

### Vectorization
```python
p_vec = NLPProcessor(vectorize="tfidf", stop_words=True)
corpus = [
    "Machine learning is fascinating.",
    "Natural language processing is a subset of AI."
]

p_vec.fit_vectorizer(corpus)
vectors = p_vec.transform_texts(corpus)
print(vectors.shape)
```

## Configuration
| Parameter | Description |
|-----------|-------------|
| `stem` | Enable stemming (PorterStemmer). |
| `lemmatize` | Enable lemmatization (WordNet). |
| `vectorize` | "tfidf", "count", or None. |
| `tokenize` | Force return of token list. |
| `remove_stopwords` | Remove English stopwords (Case-insensitive). |
| `pos_tagging` | Return (Word, Tag) tuples. |
| `ner` | Return Entity tuples (Uses Spacy). |
| `normalize` | Lowercase & remove punctuation. |
| `backend` | "nltk" (default) or "spacy". |

## License
MIT License. See [LICENSE](LICENSE) for details.

