from setuptools import setup, find_packages

setup(
    name="nlp_pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A robust NLP pipeline for stemming, lemmatization, and vectorization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp_pipeline",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "spacy",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
