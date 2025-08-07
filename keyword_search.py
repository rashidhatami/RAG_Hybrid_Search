from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas as pd
from data_process import get_data

def return_df_indices(indices_list, path_df: str):
    data = []
    df = get_data(path_df)
    for index in indices_list:
        data.append(df.loc[index].to_dict())
    return(data)

def retrive_knowledge_indices(query: str, top_k: int, bm25):
    q = word_tokenize(query.lower())
    scores = bm25.get_scores(q)
    indices = np.argsort(scores)[-top_k:][::-1].tolist()
    return indices

def retrive_knowledge(indice):
    data = return_df_indices(indice)
    docs = []
    for doc in data:
        docs.append(f"NEWS Title: {doc['title']}. Desription: {doc['description']} Date: {doc['published_at']} \nurl: {doc['url']}\n")
    return '\n\n'.join(docs)

def prepare_augmented_prompt(query, knowledge):
    prompt = "please generate a responsed based this prompt and knowledge."
    augmented_prompt = f"{prompt} \n\nprompt: {query}\n\nRelative knwoledge is:\n{knowledge}"
    return augmented_prompt


def create_bm25(docs):
    tokenized = [word_tokenize(doc.lower()) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25


def KeywordSearch(query, top_k, bm25, use_rag=True):

    if not use_rag:
        return query
    
    indices = retrive_knowledge_indices(query, top_k, bm25)
    # knoweledge = retrive_knowledge(indices)
    # prompt = prepare_augmented_prompt(query, knoweledge)

    return indices

# query = 'tell about iran'
# prompt = rag(query, top_k = 5, use_rag=True)
# index = prompt[0]
# print(prompt[0])