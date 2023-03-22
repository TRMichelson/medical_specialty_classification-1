#  Makefile for 

# target : dependencies
# 	action

# In make, $@ is a placeholder/substitution for the target name

.PHONY: all clean scispacy_model

all: 

requirements: requirements.txt
	pip install -r requirements.txt

download_model: 
	curl https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz --create-dirs -o ./nlp_models/en_core_sci_sm-0.5.1.tar.gz 

unarchive_model: ./nlp_models/en_core_sci_sm-0.5.1.tar.gz
	tar -xf ./nlp_models/en_core_sci_sm-0.5.1.tar.gz -C ./nlp_models 

install_model: 
	pip install nlp_models/en_core_sci_sm-0.5.1

scispacy_model: download_model unarchive_model install_model

data/cleaned/medical_text_clean.csv: src/clean_data.py data/raw/medical_text.csv 
	python src/clean_data.py -i data/raw/medical_text.csv -o data/cleaned/medical_text_clean.csv

delete_scispacy_model:
	rm -r nlp_models

clean: 
	rm -r data/cleaned/medical_text_clean.csv

	