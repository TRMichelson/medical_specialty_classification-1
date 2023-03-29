import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Splits dataset into train (80%) and test (20%) sets")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to input csv file to be split into training and test sets',
                        type=Path,  # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)

    
    # optional (keyword) argument with '-o1' flag to specify path to output file that will be generated
    parser.add_argument('-o1', 
                        '--output_file1', 
                        help='path to output csv file used to store training data',
                        type=Path, 
                        default=None,
                        required=True
                        )
    
    # optional (keyword) argument with '-o2' flag to specify path to output file that will be generated
    parser.add_argument('-o2', 
                        '--output_file2', 
                        help='path to output csv file used to store test data',
                        type=Path, 
                        default=None,
                        required=True
                        )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    text_df = pd.read_csv(args.input)

    # split data into train/test sets 
    train_dataset, test_dataset = train_test_split(text_df, 
                                                   test_size=0.2, 
                                                   random_state=42)

    train_csv_path = args.output_file1
    test_csv_path = args.output_file2
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)

if __name__ == "__main__":
    main()