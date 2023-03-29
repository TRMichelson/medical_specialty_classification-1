#%%
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#%%
texts = pd.read_csv("data/processed/medical_text_processed.csv")

#%%

# split into features and target
X = texts['tx_clean_ents']
y = texts['specialty_simple']

# split data into train/test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#%%
# could try tfidf for features for classification with and without cleaning the text

# instantiate TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1,3),
                             max_df=0.75, use_idf=True,
                             max_features=2000)

#%%

# apply TfidfVectorizer to X_train  
X_train_tfidf = vectorizer.fit_transform(X_train)


#%%
# inspect resulting features
feature_names = sorted(vectorizer.get_feature_names_out())
print(len(feature_names))
print(feature_names[:100])

# %%

# train classifier model on training set
clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=1)
clf.fit(X_train_tfidf, y_train)

# %%

# transform X_test with tfidf vectorizer we already fit on X_train 
X_test_tfidf = vectorizer.transform(X_test)

# predict medical specialty of X_test_tfidf
y_pred = clf.predict(X_test_tfidf)

# %%

print(classification_report(y_test, y_pred))

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
_ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}\non filtered documents"
)
# %%

# train classifier model on training set
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train_tfidf, y_train)

# %%

# transform X_test with tfidf vectorizer we already fit on X_train 
X_test_tfidf = vectorizer.transform(X_test)

# predict medical specialty of X_test_tfidf
y_pred = clf.predict(X_test_tfidf)

# %%

print(classification_report(y_test, y_pred))
# %%

# %%

