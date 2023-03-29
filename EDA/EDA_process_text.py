# tokenize
# remove stopwords
# remove punctuation
# lowercase everything
# use .isalpha() to remove numeric text 
# remove unicode characters
# stemming (porter stemmer)

#%%

import pandas as pd
import spacy
import en_core_sci_sm

# %%

texts = pd.read_csv("data/cleaned/medical_text_clean.csv")

#%%

nlp = en_core_sci_sm.load()

# %%

def clean_text(text) -> str:
    # lowercase string
    text_lower = text.lower()
    doc = nlp(text_lower)
    # extract text if the token contains all letters and is not a stop work or punctuation mark
    tokens = [token.text for token in doc if (
                                                (token.text.isalpha() == True) & 
                                                (token.is_stop == False) & 
                                                (token.is_punct == False)
                                                )]    

    # return extracted tokens as a single, joined string
    return " ".join(tokens)    

def extract_entities(doc: str) -> str:
    doc = nlp(doc)
    ent_list = [ent.text for ent in doc.ents]
    return " ".join(ent_list)

#%%
texts['tx_clean'] = texts['transcription'].apply(clean_text)

#%%
texts['tx_clean_ents'] = texts['tx_clean'].apply(extract_entities)

#%%
texts['tx_ents'] = texts['transcription'].apply(extract_entities)

