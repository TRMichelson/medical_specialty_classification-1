#%%
import duckdb
import pandas as pd

#%%
text_df = pd.read_csv("data/raw/medical_text.csv").drop('Id', axis=1)

#%%

text_df.head()

#%%
# create barchart of counts of records for each medical specialty
text_df['medical_specialty'].value_counts().plot(kind='barh')

#%%
# inspect actual counts by specialty
text_df['medical_specialty'].value_counts()

#%%
# inspect counts by specialty using SQL via duckdb 

counts_by_specialty = """
SELECT medical_specialty, COUNT(*) AS count
FROM text_df
GROUP BY medical_specialty
ORDER BY count DESC
"""

duckdb.query(counts_by_specialty).to_df()

#%% How many null (missing) transcription records?
q_count_null = """
SELECT COUNT(*) as n_null
FROM text_df
WHERE transcription IS NULL
"""

duckdb.query(q_count_null)



#%%
# remove records with missing transcription and filter to records that have a medical_specialty with at least 50 records in the dataset
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

# %%
# ILIKE is case-insensitive LIKE used by duckdb

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
# %%

rename_specialties = '''
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

text_df_renamed_specialties = duckdb.query(rename_specialties).to_df()
# %%
