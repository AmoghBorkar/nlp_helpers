#!/usr/bin/env python3
"""NLPHelpers contains various text cleaning, standardization and visualization functions.
    A lot of these are wrappers over existing packages but are intended to make working with
    text data simpler specially when working with modern models such as BERT.
    In case of issues with python 3.10, try using python 3.9.x 

Usage: Add packages to python directory or working directory 
    from nlp_helpers import NLPHelpers
    nlp_helpers = NLPHelpers()
    help(nlp_helpers) # Displays different functions available.

Author:
    Amogh Borkar - 12.07.2022
"""
from random import random
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import time
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from transformers import BertTokenizer
import copy
from nltk.stem import PorterStemmer
import re
from flair.data import Sentence
from flair.models import SequenceTagger
from wordcloud import WordCloud
import string
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from fasttext import load_model
import datetime
import lemminflect
from typing import Tuple
import wordninja
from nltk.corpus import stopwords
import nltk
import spacy

# Download added to requirements.txt.
nlp = spacy.load("en_core_web_sm")
# Check if stopwords are available, else download.
try:
    stops = set(stopwords.words("english"))
except:
    nltk.download("stopwords")


class NLPHelpers(object):
    """
    NLPHelpers contains various text cleaning, standardization and visualization functions.
    A lot of these are wrappers over existing packages but are intended to make working with
    text data simpler specially when working with modern models such as BERT.
    """

    def __init__(self) -> None:
        # Loads up the NER model for use in functions later.
        self.ner_tagger = SequenceTagger.load("flair/ner-english")

    def lemmatize_list(self, text_list: list) -> list:
        """
        This function simply applies Spacy lemmatization over a list of sentences/ single words
        Inputs:
        =======
        text_list (list): List of input strings

        Outputs:
        ========
        lemmatized_list (list): List of lemmatized output lists
        """
        lemmatized_list = []
        for text in text_list:
            doc = nlp(text)
            lemmatized_sentence = " ".join([token._.lemma() for token in doc])
            lemmatized_list.append(lemmatized_sentence)
        return lemmatized_list

    def clean_text_data(
        self,
        input_text: str,
        remove_numbers_symbols: bool = False,
        stem: bool = False,
        lemmatize: bool = False,
        min_word_len: int = 2,
        additional_stopwords: list = [],
        remove_patterns: list = [],
    ) -> str:
        """
        A generic function to perform various cleaning operations over raw text before processing.
        Inputs:
        =======
        input_text (str): Input text to be cleaned
        remove_numbers_symbols (bool): Default False. If True, only alphabets, full-stops and ampersand are preserved.
        Else numbers and symbols => .$£&'"+- are preserved.
        stem (bool): Default False. If true, stemmed text is returned.
        lemmatize (bool): Default False. If true, lemmatized text is returned.
        min_word_len (int): Default 2. Min length of words returned would be greater than this.
        additional_stopwords (list): User defined list of custom stopwords to remove. Full words are matched and removed.
        remove_patterns (list): User defined list of patterns to remove. Use this for HTML tags removal.

        Outputs:
        ========
        cleaned_string (str): Cleaned text based on the criteria defined.
        """
        # Remove the patterns
        for pattern in remove_patterns:
            input_text = input_text.replace(pattern, " ")
        # Remove symbols & numbers as specified.
        if remove_numbers_symbols == False:
            regex_string = """[^0-9a-zA-Z .$£&'"'+-]+"""
        else:
            regex_string = """[^a-zA-Z .&"""
        letters_only = re.sub(regex_string, " ", input_text)
        # Remove extra spaces and newlines
        letters_only = " ".join(
            [re.sub(r"\s+|\\n", " ", x) for x in letters_only.split()]
        )
        # Convert to lower case and split into individual words.
        words = letters_only.lower().split()
        # Remove words below the specified length
        meaningful_words = [w for w in words if len(w) > min_word_len]
        # Remove stopwords and stem if true. Else remove stopwords only.
        if stem == True:
            ps = PorterStemmer()
            meaningful_words = [ps.stem(w) for w in meaningful_words if not w in stops]
        else:
            meaningful_words = [w for w in meaningful_words if not w in stops]
        # Join back words into the string
        cleaned_string = " ".join(meaningful_words)
        # Lemmatize the string if set to true
        if lemmatize == True:
            doc = nlp(cleaned_string)
            doc = [token.lemma_ for token in doc]
            meaningful_words = [w for w in doc if not w in stops]
            cleaned_string = " ".join(meaningful_words)
        # and return the cleaned string
        return cleaned_string

    def cosine_dist_classifier(
        self,
        train_sentences_list: list,
        train_labels_list: list,
        test_sentences_list: list,
        sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        A basic function that takes in a classification dataset, a sentence transformer and applies
        labels to a dataset based on cosine similarity.

        Inputs:
        =======
        train_sentences_list (list): List of sentences that form X_train
        train_labels_list (list): List of corresonding labels that form Y_train
        test_sentences_list (list): List of sentences that form X_test
        sentence_model (str): The path/ name of the sentence embedding model to be used.
        Default model used: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        (Use sentence embedding models only. Does not work on work level embedding yet.)

        Outputs:
        ========
        predicted_class_list (list): List of predicted classes for y_test.
        cosine_similarity_list (list): List of corresponding cosine similarities of the best match.
        This can be used to set thresholds for similarity.
        """
        # Load the sentence model
        sentence_model = SentenceTransformer(sentence_model)
        train_emb_list = [[sentence_model.encode(x)] for x in train_sentences_list]
        predicted_class_list = []
        cosine_similarity_list = []
        for row in test_sentences_list:
            emb_to_check = [sentence_model.encode(row)]
            emb_cosine_similarity_list = [
                cosine_similarity(emb_to_check, x) for x in train_emb_list
            ]
            predicted_class_list.append(
                train_labels_list[np.argmax(emb_cosine_similarity_list)]
            )
            cosine_similarity_list.append(np.max(emb_cosine_similarity_list))
        return predicted_class_list, cosine_similarity_list


    def get_ner_tags(
        self,
        text: str,
    ):
        """
        This function takes a sentence and returns a list of NER tags within it.

        Inputs:
        =======
        text (str): Input text

        Outputs:
        ========
        ner_list (list): A list of NER text, tags, probabiliti found in the sentence delimited by --
        """
        sentence = Sentence(text)
        self.ner_tagger.predict(sentence)
        ner_list = []
        for entity in sentence.get_spans("ner"):
            ner_list.append(str(entity.text)+"--"+str(entity.tag)+"--"+str(entity.score))
        return ner_list

    def autorun_bertopic(self,docs_list:list, num_topics = "auto") -> Tuple[list, pd.DataFrame]:
        """
        A basic wrapper over bertiopic algorithm to run it in standard settings. This also tries to wrap the
        recommendations mentioned on this page to run bertopic without much tweaking 
        - https://maartengr.github.io/BERTopic/faq.html#why-are-the-results-not-consistent-between-runs
        
        Inputs:
        =======
        docs_list(list): List of input text data.
        num_topics (str/int): Number of topipcs if specified. Else auto.
        
        Outputs: 
        ========
        topics (list): Topic Number for each element in input data.
        """
        # Initialize params
        vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words="english",min_df=1)
        embedding_model  = "sentence-transformers/all-MiniLM-L6-v2"
        # Set the params based on the size of input data.
        # Check for large dataset
        if len(docs_list) > 50000: # Define params for large datasets.
            n_gram_range = [1,2]
            nr_topics = num_topics
            top_n_words = 15
            min_topic_size = 10
            # Define HDBSCAN outside BERTopic to reduce number of outliers.
            hdbscan_model = HDBSCAN(min_cluster_size=10,prediction_data=True,min_samples=5, core_dist_n_jobs=-1)
            # Define the UMAP Model
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
            low_memory = True
            calculate_probabilities = False
        elif len(docs_list) < 2000: # Tweak params for very small datasets.
            n_gram_range = [1,2]
            # For very small datasets, set nr_topics to None to skip MMR which throws errors.
            if len(docs_list) <= 500:
                nr_topics = None
            else:
                nr_topics = num_topics
            # Set values of other params
            top_n_words = 15
            min_topic_size = 3
            # Define the UMAP model for small dataset
            umap_model = UMAP(n_neighbors=3, n_components=3, min_dist=0.0, metric="cosine", random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True, min_samples=5)
            low_memory = False
            calculate_probabilities = True
        else:
            n_gram_range = [1,2]
            nr_topics = num_topics
            # Set values of other params
            top_n_words = 15
            min_topic_size = 5
            umap_model = UMAP(n_neighbors=15, n_components=5, 
                  min_dist=0.0, metric='cosine', random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True, min_samples=5)
            low_memory = False
            calculate_probabilities = True
        # Initialize the model
        topic_model = BERTopic(embedding_model=embedding_model,
        n_gram_range=n_gram_range, nr_topics=nr_topics, top_n_words=top_n_words, 
        calculate_probabilities=calculate_probabilities,vectorizer_model=vectorizer_model, 
        min_topic_size=min_topic_size, low_memory=low_memory, verbose=True, umap_model=umap_model, hdbscan_model=hdbscan_model)
        # Run BERTopic: Fit the model
        topics, probs = topic_model.fit_transform(docs_list)
        # Build the topic model info dataframe
        topic_model_info_df = topic_model.get_topic_info()
        topic_model_info_df["topic_keywords"] = ""
        for index, row in topic_model_info_df.iterrows():
            topic_num = row["Topic"]
            topics_list = topic_model.get_topic(topic_num)
            topics_list = [x[0] for x in topics_list]
            topic_model_info_df.loc[index, "topic_keywords"] = str(topics_list)
        return topics, topic_model_info_df



