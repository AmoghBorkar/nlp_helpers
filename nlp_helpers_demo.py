#!/usr/bin/env python3
'''Prints Basic Usage and demonstrates usage of functions from the module
   nlp_helpers

Usage:
    ./nlp_helpers_demo.py

Author:
    Amogh Borkar - 17.07.2022
'''

from nlp_helpers import NLPHelpers
from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

# Instantiate
nlp_helpers = NLPHelpers()

#Examples of text cleaning
print(" Examples of text cleaning & NER:")
print("==================================")
sentence_to_clean = """<h3 class="story-title">
								Japan PM Kishida COVID positive, cancels African development conference trip</h3>
							</a>
			        <div class="contributor"></div>
			        <p>    Japanese Prime Minister Fumio Kishida has tested positive for COVID-19, forcing him to cancel a planned trip to Tunisia to attend a key conference on African development, a person close to him said on Sunday.</p>
					<time class="article-time">
							<span class="timestamp">7:55am EDT</span>"""

remove_patterns = ['<h3 class="story-title">','<div class="contributor"></div>','<time class="article-time">','<span class="timestamp">7:55am EDT</span>', '<p>']

print("-------------------")
print(f"Input Sentence:")
print(sentence_to_clean)
print("-------------------")
print(f"With nlp_helpers.clean_text_data(sentence_to_clean):")
print(nlp_helpers.clean_text_data(sentence_to_clean))
print("-------------------")
print(f'''With nlp_helpers.clean_text_data(sentence_to_clean) and remove_patterns = {remove_patterns}:''')
print(nlp_helpers.clean_text_data(sentence_to_clean,remove_patterns=remove_patterns))
print("-------------------")
print("With stemming:")
print(nlp_helpers.clean_text_data(sentence_to_clean,remove_patterns=remove_patterns, stem=True))
print("-------------------")
print("Lemmatized:")
print(nlp_helpers.clean_text_data(sentence_to_clean,remove_patterns=remove_patterns, lemmatize=True))
print(" NER Tags from lemmatized string:")
print(nlp_helpers.get_ner_tags(nlp_helpers.clean_text_data(sentence_to_clean,remove_patterns=remove_patterns, lemmatize=True)))



# print(" Examples of Topic Analysis:")
# print("==================================")
# topics, topic_model_info_df = nlp_helpers.autorun_bertopic(docs)