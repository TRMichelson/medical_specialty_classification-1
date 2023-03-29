# Makefile for medical_specialty_classification demo project
# Note that make commands must be run from project root folder (where makefile should be located)

# target : dependencies
# 	action



.PHONY: all clean scispacy_model 

.ONESHELL:
# run whole source code pipeline to build files listed here
all: data/cleaned/medical_text_clean.csv data/processed/medical_text_processed.csv data/split/medical_text_train.csv data/split/medical_text_test.csv ml_models/rfc_tfidf_tx_clean_ents.joblib reports/clf_report_rfc_tfidf_text_clean_ents.csv

# install required python packages for project
requirements: requirements.txt
	pip install -r requirements.txt

# download pre-trained nlp model
download_scispacy_model: 
	curl https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz -o ./nlp_models/en_core_sci_sm-0.5.1.tar.gz 

# unarchive the nlp model
unarchive_scispacy_model: ./nlp_models/en_core_sci_sm-0.5.1.tar.gz
	tar -xf ./nlp_models/en_core_sci_sm-0.5.1.tar.gz -C ./nlp_models 

# install the nlp model as a python package
install_scispacy_model: 
	pip install nlp_models/en_core_sci_sm-0.5.1

# chain commands to download, unarchive and install the nlp model
scispacy_model: download_scispacy_model unarchive_scispacy_model install_scispacy_model

# delete downloaded spacy/scispacy models
delete_nlp_models:
	rm -r nlp_models/*.*

# ------- ML PIPELINE STAGES FROM src FOLDER BELOW ---------------- 

# clean the categories of the raw data
data/cleaned/medical_text_clean.csv: src/clean_data.py data/raw/medical_text.csv 
	python src/clean_data.py -i data/raw/medical_text.csv -o data/cleaned/medical_text_clean.csv

# use nlp model to do text cleaning (preprocessing) on the texts 
data/processed/medical_text_processed.csv: src/process_text.py data/cleaned/medical_text_clean.csv
	python src/process_text.py -i data/cleaned/medical_text_clean.csv -o data/processed/medical_text_processed.csv

# split data into train and test data
data/split/medical_text_train.csv data/split/medical_text_test.csv: src/split_data.py data/processed/medical_text_processed.csv
	python src/split_data.py -i data/processed/medical_text_processed.csv -o1 data/split/medical_text_train.csv -o2 data/split/medical_text_test.csv

# train and tune classifier and store trained model
ml_models/rfc_tfidf_tx_clean_ents.joblib: src/train_classifier.py data/split/medical_text_train.csv
	python src/train_classifier.py -i data/split/medical_text_train.csv -o ml_models/rfc_tfidf_tx_clean_ents.joblib 

# generate classification report on test data
reports/clf_report_rfc_tfidf_text_clean_ents.csv: src/evaluate_classifier.py ml_models/rfc_tfidf_tx_clean_ents.joblib data/split/medical_text_test.csv
	python src/evaluate_classifier.py -i1 ml_models/rfc_tfidf_tx_clean_ents.joblib -i2 data/split/medical_text_test.csv -o reports/clf_report_rfc_tfidf_text_clean_ents.csv

# ---- END OF PIPELINE STAGES -----------------------------------------

# delete output files that result from running project pipeline
clean: 
	rm -r data/cleaned/medical_text_clean.csv
	rm -r data/processed/medical_text_processed.csv
	rm -r data/split/medical_text_train.csv data/split/medical_text_test.csv 
	rm -r ml_models/rfc_tfidf_tx_clean_ents.joblib
	rm -r reports/clf_report_rfc_tfidf_text_clean_ents.csv
	

	