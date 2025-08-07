import pandas as pd
from data_process import extract_doc, get_data
from keyword_search import KeywordSearch, create_bm25
from semantic_search import embedding_model, embed_doc, save_embedding, load_embedding, embed_query, retrive_indices


path = 'news_data.csv'
df = get_data(path)
docs = extract_doc(df)
query = "What are the recent news about Iran?"


# -------------------- keyword search ----------------------
bm25 = create_bm25(docs)
keyword_indices = KeywordSearch(query, 5, bm25)
print("keyword indices are: ", keyword_indices)


# --------------------- semantic search ---------------------
model = embedding_model(model='all-MiniLM-L6-v2')
# embedding = embed_doc(docs, model)
# save_embedding(embedding)
embedding = load_embedding('embedding.joblib')
embed_query = embed_query(query, model)
semantic_indices = retrive_indices(embed_query, embedding, 5)

print("semantic indices are: ", semantic_indices)

