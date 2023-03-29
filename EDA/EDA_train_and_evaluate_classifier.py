
#%%
import argparse
from joblib import dump, load
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import (cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold)

#%%

def tune_params_logreg_tfidf(X, y):
    ''''''
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, max_df=0.75)),
            ('logreg', LogisticRegression(solver='saga'))
            ])

    params = {
        'tfidf__max_features': range(500, 5001),
        'tfidf__ngram_range': ((1,1), (1,2), (1,3)),
        'logreg__penalty': ['l1', 'l2', 'elasticnet'],
        'logreg__C': uniform(0, 10)
    }

    # create a RandomizedSearchCV object
    search = RandomizedSearchCV(pipeline, 
                                params, 
                                scoring=make_scorer(f1_score, average='micro'), 
                                n_iter=3, 
                                cv=5, 
                                random_state=42)

    # fit the RandomizedSearchCV object to the data
    search.fit(X, y)

    # print the best hyperparameters and best score
    print("Best hyperparameters: ", search.best_params_)
    print("Best score: ", search.best_score_)

    return(search)

#%%

def tune_params_rfc_tfidf(X, y):
    ''''''
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, max_df=0.75)),
            ('rfc', RandomForestClassifier())
            ])

    params = {
        'tfidf__max_features': range(500, 5001),
        'tfidf__ngram_range': ((1,1), (1,2), (1,3)),
        'rfc__n_estimators': range(30, 301),
        'rfc__max_features': ('sqrt', 'log2')
    }

    # create a RandomizedSearchCV object
    search = RandomizedSearchCV(pipeline, 
                                params, 
                                scoring=make_scorer(f1_score, average='micro'), 
                                n_iter=2, 
                                cv=5, 
                                refit=True,
                                random_state=42)

    # fit the RandomizedSearchCV object to the data
    search.fit(X, y)

    # print the best hyperparameters and best score
    print("Best hyperparameters: ", search.best_params_)
    print("Best score: ", search.best_score_)

    return(search)

#%%
'''
from joblib import dump, load
svc = svm.SVC() # Probably not what you are using, but just as an example
gcv = GridSearchCv(svc, parameters, refit=True) 
gvc.fit(X, y)
estimator = gcv.best_estimator_
dump(estimator, "your-model.joblib")
# Somewhere else
estimator = load("your-model.joblib")
'''


#%%

text_df = pd.read_csv("data/split/medical_text_train.csv")

# split into features and target
X_train = text_df['tx_clean_ents']
y_train = text_df['specialty_simple']

#%%
#logreg_tfidf_search = tune_params_logreg_tfidf(X, y)


#%%
rfc_param_search = tune_params_rfc_tfidf(X_train, y_train)



#%%
rfc_param_search.best_score_

#%%

dump(rfc_param_search.best_estimator_, 'ml_models/rfc_tfidf_tx_clean_ents.joblib')

#%%
clf = load('ml_models/rfc_tfidf_tx_clean_ents.joblib')

#%%

# Predictions/Performance on test set

text_test_df = pd.read_csv("data/split/medical_text_test.csv")

# split into features and target
X_test = text_test_df['tx_clean_ents']
y_test = text_test_df['specialty_simple']

clf_report_dict = classification_report(y_test, clf.predict(X_test), output_dict=True)

clf_report_df = pd.DataFrame(clf_report_dict).transpose()

#clf_report_df.to_csv(args.output_file2)



