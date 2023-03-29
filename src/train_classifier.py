#%%
import argparse
from joblib import dump, load
from pathlib import Path

import pandas as pd
from scipy.stats import uniform
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV


#%%

def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Trains and tunes hyperparams of a classifier model on text_clean_ents and medical_specialty labels using RandomizedSearchCV")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to input csv file with training data containing text_clean_ents and specialty_simple columns',
                        type=Path,      # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to output joblib file that stores trained classifier models with best params',
                        type=Path,
                        default=None,
                        required=True)

    args = parser.parse_args()
    return args


def tune_params_logreg_tfidf(X, y):
    '''Train and tune tfidf and LogisticRegression pipeline on training data'''
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


def tune_params_rfc_tfidf(X, y):
    '''Train and tune tfidf and RandomForestClassifier pipeline on training data'''
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

def main():

    # get command-line arguments
    args = parse_args()

    # load dataframe from csv containing texts
    text_train_df = pd.read_csv(args.input)

    # split training data into features and target
    X_train = text_train_df['tx_clean_ents']
    y_train = text_train_df['specialty_simple']

    # train and tune RandomForestClassifier and return best scoring model
    rfc_param_search = tune_params_rfc_tfidf(X_train, y_train)

    # store the model with the highest score
    dump(rfc_param_search.best_estimator_, args.output)


if __name__ == "__main__":
    main()
