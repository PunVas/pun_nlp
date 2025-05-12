# NLPProcessor

## Overview
NLPProcessor is an automated, adaptive NLP pipeline that dynamically handles:
- **Tokenization** (Word & Sentence)
- **Stopword Removal**
- **POS Tagging**
- **Named Entity Recognition (NER)**
- **Text Normalization** (Lowercasing, Punctuation Removal, etc.)
- **Stemming & Lemmatization** (via NLTK or spaCy)
- **Vectorization** (TF-IDF or Count Vectorizer)
- **Dependency Management** (Auto-installs missing libraries.)
- **Support for 2D Text Arrays** (Processes lists of lists of text.)
- **Exception-Free Execution** (Handles API changes without breaking.)

## Features
- **Automated dependency installation**
- **Works with both NLTK and spaCy**
- **Vectorization support using scikit-learn**
- **Handles single strings and 2D arrays**
- **No human intervention required**

## Installation
Run the following command to install missing dependencies:
```bash
pip install pun_nlp
```

## Usage
### Import and Initialize
```python
from pun_nlp import NLPProcessor

processor = NLPProcessor(stem=True, lemmatize=True, vectorize="tfidf", backend="spacy")
```

### Process a Single Text
```python
output = processor.process("running jumped swimming")
print(output)
```

### Process a 2D Array of Text
```python
input_texts = [
    ["I am running", "He is jumping"],
    ["They are swimming", "Dogs are barking"]
]
output = processor.process(input_texts)
print(output)
```

### Customization Options
| Parameter | Description |
|-----------|-------------|
| `stem` | Enable stemming (default: `False`) |
| `lemmatize` | Enable lemmatization (default: `False`) |
| `vectorize` | Choose "tfidf", "count", or `None` (default: `None`) |
| `tokenize` | Enable word/sentence tokenization (default: `False`) |
| `remove_stopwords` | Remove stopwords (default: `False`) |
| `pos_tagging` | Enable Part-of-Speech tagging (default: `False`) |
| `ner` | Enable Named Entity Recognition (default: `False`) |
| `normalize` | Lowercase and remove punctuation (default: `False`) |
| `backend` | Choose "nltk" or "spacy" (default: "nltk") |

### Check Supported Vectorizers
```python
print(NLPProcessor.supported_vectorizers())  # ['tfidf', 'count']
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

