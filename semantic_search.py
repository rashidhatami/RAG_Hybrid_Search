from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np


def embedding_model(model='all-MiniLM-L6-v2'):
    model_emb = SentenceTransformer(model)
    return model_emb

def embed_doc(knowledge, model):
    embedding = [model.encode(doc.lower()) for doc in knowledge]
    return embedding

def save_embedding(embedding):
    joblib.dump(embedding, 'embedding.joblib')

def load_embedding(path):
    return joblib.load(path)


def embed_query(query: str, model):
    return model.encode(query.lower())

def retrive_indices(embed_query, embedding, top_k=3):

    scores = cosine_similarity([embed_query], embedding)[0]
    indices = np.argsort(scores)[::-1]

    return indices[:top_k]



# model = embedding_model()
# docs = extract_doc()
# embedding = embed_doc(docs, model)
# save_embedding()
# embedding = load_embedding('embedding.joblib')

# query = "Gaza"
# embed_query = embed_query(query, model)

# retrive_indices(embed_query, embedding, 5)