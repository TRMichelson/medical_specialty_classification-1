
# adapted from https://umap-learn.readthedocs.io/en/latest/document_embedding.html

#%%
import pandas as pd
import umap
import umap.plot

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE
output_notebook(resources=INLINE)

#%%
texts = pd.read_csv('data/processed/medical_text_preprocessed.csv')
# %%

vectorizer = CountVectorizer(min_df=5, stop_words='english')
word_doc_matrix = vectorizer.fit_transform(texts['tx_clean_ents'])
# %%
embedding = umap.UMAP(n_components=2, metric='hellinger').fit(word_doc_matrix)


# %%
f = umap.plot.points(embedding, labels=texts['specialty_simple'])

# %%
tfidf = TfidfVectorizer(min_df=5, stop_words='english')
tfidf_word_doc_matrix = tfidf.fit_transform(texts['tx_clean_ents'])
# %%
tfidf_embedding = umap.UMAP(n_components=2, metric='hellinger').fit(tfidf_word_doc_matrix)
# %%
f_tfidf = umap.plot.points(tfidf_embedding, labels=texts['specialty_simple'])
# %%
