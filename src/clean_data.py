import duckdb
import pandas as pd

def main():

    text_df = pd.read_csv("data/raw/medical_text.csv").drop('Id', axis=1)

    # remove records with missing transcription 
    # and filter to records that have a medical_specialty with at least 50 records in the dataset
    select_specialties_atleast_50_recs = """
    WITH fifty_or_more_records AS (
    SELECT medical_specialty, COUNT(*) AS count
    FROM text_df
    GROUP BY medical_specialty
    HAVING count >=50
    )
    SELECT *
    FROM text_df
    WHERE medical_specialty IN (
        SELECT medical_specialty
        FROM fifty_or_more_records)
        AND transcription IS NOT NULL
    """

    text_df_filtered = duckdb.query(select_specialties_atleast_50_recs).to_df()

    text_df_filtered.to_csv("data/cleaned/medical_text_clean.csv", index=False)

if __name__ == "__main__":
    main()