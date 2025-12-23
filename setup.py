from setuptools import setup, find_packages


setup(
    name="pun_nlp",
    version="0.0.9",
    author="Puneet Vaswani",
    author_email="vaswaniusham2212@gmail.com",
    description="A robust NLP pipeline for stemming, lemmatization, and vectorization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PunVas/pun_nlp",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "spacy",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
