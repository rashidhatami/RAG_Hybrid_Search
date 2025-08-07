import pandas as pd


def get_data(path: str):
    return pd.read_csv(path)

def extract_doc(dataframe):
    docs = [f"{row['title']}. {row['description']} {row['published_at']}" for _,row in dataframe.iterrows()]
    return docs

