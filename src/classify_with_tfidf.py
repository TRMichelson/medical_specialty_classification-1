#%%
import argparse
from joblib import dump
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Cleans input csv file")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to input file',
                        type=Path,      # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to output file',
                        type=Path,
                        default=None,
                        required=False)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    text_df = pd.read_csv(args.input)

    # split into features and target
    X = text_df['tx_clean_ents']
    y = text_df['specialty_simple']

    # split data into train/test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # instantiate TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1,3),
                             max_df=0.75, use_idf=True,
                             max_features=2000)


    # fit TfidfVectorizer to X_train and generate word-document tfidf matrix
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # train classifier model on training set
    clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=1)
    clf.fit(X_train_tfidf, y_train)

    # Save model
    dump(clf, "ml_models/logreg_tfidf_tx_clean_ents.joblib")

    # transform X_test with tfidf vectorizer we already fit on X_train 
    X_test_tfidf = vectorizer.transform(X_test)

    # predict medical specialty of X_test_tfidf
    y_pred = clf.predict(X_test_tfidf)

    # generate classification report
    clf_report_dict = classification_report(y_test, y_pred, output_dict=True)

    # save classification report

    clf_report_df = pd.DataFrame(clf_report_dict).transpose()

    clf_report_df.to_csv('reports/clf_report_logreg_tfidf_tx_clean_ents.csv')

    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    #ax.xaxis.set_ticklabels(target_names)
    #ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}\non filtered documents"
    )
    """


if __name__ == "__main__":
    main()