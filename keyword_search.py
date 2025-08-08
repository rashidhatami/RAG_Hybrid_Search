from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas as pd
from data_process import get_data



def retrive_knowledge_indices(query: str, top_k: int, bm25):
    q = word_tokenize(query.lower())
    scores = bm25.get_scores(q)
    indices = np.argsort(scores)[-top_k:][::-1].tolist()
    return indices

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