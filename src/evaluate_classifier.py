import argparse
from joblib import load
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Evaluate performance on the test data using the trained model")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i1',
                        '--input1',
                        help='Path to stored classifier model',
                        type=Path,  # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)
    
    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i2',
                        '--input2',
                        help='Path to test data with specialty_simple and text_clean_ents columns',
                        type=Path,  # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)
    
    # optional (keyword) argument with '-o1' flag to specify path to output file that will be generated
    parser.add_argument('-o', 
                        '--output_file', 
                        help='path to where to write classification report',
                        type=Path, 
                        default=None,
                        required=True
                        )

    args = parser.parse_args()
    return args


def main():

    # get command-line arguments
    args = parse_args()

    # load dataframe from csv containing texts
    text_test_df = pd.read_csv(args.input2)

    # split test data into features and target
    X_test = text_test_df['tx_clean_ents']
    y_test = text_test_df['specialty_simple']

    # load the classifier model from file
    clf = load(args.input1)

    # apply model to test data and generate classification report
    clf_report_dict = classification_report(y_test, clf.predict(X_test), output_dict=True)

    # convert classification report to dataframe
    clf_report_df = pd.DataFrame(clf_report_dict).transpose()

    # write dataframe to csv 
    clf_report_df.to_csv(args.output_file)


if __name__ == "__main__":
    main()