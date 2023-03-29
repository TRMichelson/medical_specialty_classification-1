import argparse
import pandas as pd
from pathlib import Path
import en_core_sci_sm


def parse_args():
    """ Get command line arguments """

    parser = argparse.ArgumentParser(description="Uses spacy nlp model to do text cleaning/preprocessing on texts in transcription columns of input csv file and stores cleaned/preprocessed text as added columns in output csv file")

    # optional (keyword) argument with '-i' flag to accept pre-existing input file(s)
    parser.add_argument('-i',
                        '--input',
                        help='Path to input csv file containing medical_specialty and transcription columns',
                        type=Path,      # Note that this requires import as follows “from pathlib import Path”
                        default=None,
                        required=True)

    # optional (keyword) argument with '-o' flag to specify path to output file(s) that will be generated
    parser.add_argument('-o',
                        '--output',
                        help='Path to output csv file with added columns containing cleaned/preprocessed texts',
                        type=Path,
                        default=None,
                        required=True)

    args = parser.parse_args()
    return args


def clean_text(text: str, nlp_model) -> str:
    ''' Clean/preprocess text using passed spacy language model'''

    # lowercase string
    text_lower = text.lower()

    doc = nlp_model(text_lower)
    
    # extract text if the token contains all letters and is not a stop work or punctuation mark
    tokens = [token.text for token in doc if (
                                                (token.text.isalpha() == True) & 
                                                (token.is_stop == False) & 
                                                (token.is_punct == False)
                                                )]    

    # return extracted tokens as a joined string
    return " ".join(tokens)    


def extract_entities(doc: str, nlp_model) -> str:
    '''Extracts the text for entities identified in text (doc) using passed spacy language model'''

    # apply spacy nlp model to text (doc)
    doc = nlp_model(doc)
    # extract text for entities found in document 
    ent_list = [ent.text for ent in doc.ents]

    # return text for extracted entities as a joined string
    return " ".join(ent_list)


def main():

    # get command-line arguments
    args = parse_args()

    # load dataframe from csv containing texts
    text_df = pd.read_csv(args.input)

    # load spacy (scispacy) model
    nlp = en_core_sci_sm.load()

    # clean (preprocess) transcription texts and store into new col
    text_df['tx_clean'] = text_df['transcription'].apply(clean_text, nlp_model=nlp)

    # extract entities from clean transcription texts and store into new col
    text_df['tx_clean_ents'] = text_df['tx_clean'].apply(extract_entities, nlp_model=nlp)
    
    # write df with newly added text cols to file 
    text_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()