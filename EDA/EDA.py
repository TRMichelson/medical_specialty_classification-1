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

#%%
# filter to records that have a medical_specialty with at least 50 records in the dataset
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
"""

text_df_filtered = duckdb.query(select_specialties_atleast_50_recs).to_df()
# %%
