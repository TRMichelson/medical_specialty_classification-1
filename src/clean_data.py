import argparse
import duckdb
from pathlib import Path
import pandas as pd

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
                        required=True)

    args = parser.parse_args()
    return args

def main():

    #text_df = pd.read_csv("data/raw/medical_text.csv").drop('Id', axis=1)

    args = parse_args()

    text_df = pd.read_csv(args.input).drop('Id', axis=1)

    # remove records with missing transcription 
    # and filter to records that have a medical_specialty with at least 50 records in the dataset
    # only select transcription and medical specialty columns
    select_specialties_atleast_50_recs = """
    WITH fifty_or_more_records AS (
    SELECT medical_specialty, COUNT(*) AS count
    FROM text_df
    GROUP BY medical_specialty
    HAVING count >=50
    )
    SELECT 
        transcription, 
        medical_specialty
    FROM text_df
    WHERE medical_specialty IN (
        SELECT medical_specialty
        FROM fifty_or_more_records)
        AND transcription IS NOT NULL
    """

    text_df_filtered = duckdb.query(select_specialties_atleast_50_recs).to_df()

    # remove overly broad or general medical note types
    filter_specialties = """
    SELECT * 
    FROM text_df_filtered
    WHERE 
        medical_specialty NOT ILIKE '%surgery%' 
        AND medical_specialty NOT ILIKE '%consult%'
        AND medical_specialty NOT ILIKE '%soap%'
        AND medical_specialty NOT ILIKE '%discharge%'
    """                 

    text_df_reduced_specialties = duckdb.query(filter_specialties).to_df()


    # Create new column with simplified specialty name
    simplify_specialty_names = '''
    SELECT
        transcription,
        medical_specialty,
    CASE 
        WHEN medical_specialty ILIKE '%cardiovascular%' THEN 'cardio'
        WHEN medical_specialty ILIKE '%general%' THEN 'gen_med'
        WHEN medical_specialty ILIKE '%orthopedic%' THEN 'ortho'
        WHEN medical_specialty ILIKE '%gastroenterology%' THEN 'gastro'
        WHEN medical_specialty ILIKE '%pain management%' THEN 'pain'
        WHEN medical_specialty ILIKE '%radiology%' THEN 'rad'
        WHEN medical_specialty ILIKE '%hematology%' THEN 'hem_onc'
        WHEN medical_specialty ILIKE '%neurology%' THEN 'neuro'
        WHEN medical_specialty ILIKE '%nephrology%' THEN 'neph'
        WHEN medical_specialty ILIKE '%ENT%' THEN 'ent'
        WHEN medical_specialty ILIKE '%ophthal%' THEN 'opth'
    END AS specialty_simple
    FROM text_df_reduced_specialties
    '''

    text_df_renamed_specialties = duckdb.query(simplify_specialty_names).to_df()

    text_df_renamed_specialties.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()