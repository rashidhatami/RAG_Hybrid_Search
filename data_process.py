import pandas as pd


def get_data(path: str):
    return pd.read_csv(path)

def extract_doc(dataframe):
    docs = [f"{row['title']}. {row['description']}" for _,row in dataframe.iterrows()]
    return docs

#------------------Retriev indices -----------------
def return_df_indices(indices_list, df):
    data = []
    # df = get_data(path_df)
    for index in indices_list:
        data.append(df.loc[index].to_dict())
    return(data)

def retrive_knowledge(indice, df):
    data = return_df_indices(indice, df)
    docs = []
    for doc in data:
        docs.append(f"NEWS Title: {doc['title']}. Desription: {doc['description']} Date: {doc['published_at']} \nurl: {doc['url']}\n")
    return '\n\n'.join(docs)

def prepare_augmented_prompt(query, knowledge):

    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"Query: {query}\n"
        f"2024 News: {knowledge}"
    )
    return prompt


def RRF(keyword_indices, semantic_indices, top_k=3, k=60):

    scores = {}
    for rank, item in enumerate(keyword_indices, start=1):
        scores[item] = scores.get(item, 0) + 1/(k+rank)

    for rank, item in enumerate(semantic_indices, start=1):
        scores[item] = scores.get(item, 0) + 1/(k+rank)

    scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
    indices = [int(indice) for indice, _ in scores]

    return indices[:top_k]