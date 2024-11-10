import os
import pandas as pd
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

source_path = "Accounts"
vector_path = "Vectors"
#gpt-3.5-turbo,text-embedding-3-small,text-embedding-3-large
model_name = "text-embedding-3-small"
prompt1 = "Which customers will default this month?"
prompt2 = "Which customers will have lower FICO this month?"
prompt3 = "Which customers will have delinquency more than 0 this month?"

model_input_params = {
    "temperature": 0.0,
    "max_tokens": 300,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "model": model_name,
}

def load_cust_vector(fname: str) -> dict[tuple[int, int, int]]:
    df = pd.read_csv(fname, header=0)
    print(df.columns)
    max_dim = max([int(c) for c in df.columns if c != "idx"])
    return {
        #(r.idx): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }


def create_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    #logic for processing and extracting text
    return ""


def query_model(query: str, df: pd.DataFrame, cust_embeddings: dict[tuple[int, int, int]]) -> str:
    prompt = create_prompt(query, cust_embeddings, df)
    #print(prompt)
    #quit()

    response = client.chat.completions.creat(prompt=prompt, **model_input_params)
    print(response["choices"])
    return response["choices"][0]["text"].strip(" \n")



print('loading ...')
cust_embeddings = load_cust_vector(vector_path + "/combined_cust_vector.csv")
print('loading done.')


response = query_model(prompt1, df, cust_embeddings)
print(response)

'''
response = query_model(prompt2, df, cust_embeddings)
print(response)

response = query_model(prompt3, df, cust_embeddings)
print(response)
'''
