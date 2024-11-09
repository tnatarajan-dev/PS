import os
import pandas as pd
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

source_path = "Accounts"
vector_path = "Vectors"
#gpt-3.5-turbo,text-embedding-3-small,text-embedding-3-large
model_name = "text-embedding-3-small"


def get_embedding_model(text: str, model: str = model_name) -> list[float]:
    result = client.embeddings.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def create_cust_vector(df: pd.DataFrame)-> dict[tuple[int, int, int]]:
    return {
        idx: get_embedding_model(str(r.FICO) + "-" + str(r.Delinquency)) for idx, r in df.iterrows()
    }

with os.scandir(source_path) as dir:
    for file in dir:
        print(file)
        if file.name.endswith(".xlsx") and file.is_file():
            print(file.name, file.path)
            
            #cust_df = pd.read_excel(file, usecols="B,G,I,K,P,Q,V,W")
            cust_df = pd.read_excel(file, usecols="B,N,R")
            print(cust_df)
            
            cust_vector = create_cust_vector(cust_df)
            entry_vector = list(cust_vector.items())[0]
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"{entry_vector[0]} : {entry_vector[1][:2]}... ({len(entry_vector[1])} entries)")
            pd.DataFrame(cust_vector).T.to_csv(vector_path + "/" + file.name + "_vector.csv")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


