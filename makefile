#  Makefile for 

# target : dependencies
# 	action

# In make, $@ is a placeholder/substitution for the target name

.PHONY: all clean scispacy_model

all: data/cleaned/medical_text_clean.csv data/processed/medical_text_processed.csv train_classifier

requirements: requirements.txt
	pip install -r requirements.txt

download_scispacy_model: 
	curl https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz --create-dirs -o ./nlp_models/en_core_sci_sm-0.5.1.tar.gz 

unarchive_scispacy_model: ./nlp_models/en_core_sci_sm-0.5.1.tar.gz
	tar -xf ./nlp_models/en_core_sci_sm-0.5.1.tar.gz -C ./nlp_models 

install_scispacy_model: 
	pip install nlp_models/en_core_sci_sm-0.5.1

scispacy_model: download_scispacy_model unarchive_scispacy_model install_scispacy_model

data/cleaned/medical_text_clean.csv: src/clean_data.py data/raw/medical_text.csv 
	python src/clean_data.py -i data/raw/medical_text.csv -o data/cleaned/medical_text_clean.csv

data/processed/medical_text_processed.csv: src/process_text.py data/cleaned/medical_text_clean.csv
	python src/process_text.py -i data/cleaned/medical_text_clean.csv -o data/processed/medical_text_processed.csv

train_classifier:
	python src/classify_with_tfidf.py -i data/processed/medical_text_processed.csv

delete_scispacy_model:
	rm -r nlp_models

clean: 
	rm -r data/cleaned/medical_text_clean.csv
	rm -r data/processed/medical_text_processed.csv
	rm -r reports/clf_report_logreg_tfidf_tx_clean_ents.csv
	rm -r ml_models/logreg_tfidf_tx_clean_ents.joblib

	