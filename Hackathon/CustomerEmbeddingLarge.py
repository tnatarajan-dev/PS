import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
# Ignore all warnings
import warnings
warnings.simplefilter('ignore')


# Setting up the deployment name
model_name = "text-embedding-3-large"

# The base URL for your Azure OpenAI resource
#"https://genai-openai-profitsentinel.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
openai_api_base = "genai-openai-profitsentinel.openai.azure.com"

# The API key for your Azure OpenAI resource.
openai_api_key = "33d4c5bf7f124d05b6c20a849864a752"

# Currently OPENAI API have the following versions available: 2022-12-01
openai_api_version = "2023-05-15"

# Request URL
api_url = f"https://{openai_api_base}/openai/deployments/{model_name}/embeddings?api-version={openai_api_version}"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_embeddings(text, model=model_name):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = generate_embeddings(
        user_query,
        model=model_name
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res


# Read customer input data
#df = pd.read_excel(os.path.join(os.getcwd(),'evaluator_data_2.xlsx'))
df = pd.read_csv(os.path.join(os.getcwd(),'evaluator_data_2.csv'))
#print(df)

# Interested columns from the spreadsheet
df_customers = df[['customer', 'riskcategory', 'analysis']]
#print(df_customers)

client = AzureOpenAI(
    api_key = openai_api_key,
    api_version = openai_api_version,
    azure_endpoint = api_url
)

df_filtered_customers = df_customers[df['riskcategory'] == 'Very High Risk']
df_filtered_customers['ada_v2'] = df_filtered_customers['analysis'].apply(lambda x : generate_embeddings (x, model = model_name))

print("\n")
print("\n")
query1 = "Can I get information on customer who have financial distress?"
print(f"Query: {query1}\n")
res1 = search_docs(df_filtered_customers, query1, top_n=2)
print(res1)
print(type(res1))
print("\n")
print("\n")

query2 = "Filter results based on customers who have financial distress"
print(f"Query: {query2}\n")
res2 = search_docs(df_filtered_customers, query2, top_n=2)
print(res2)
print(type(res2))
print("\n")
print("\n")

query3 = "Can I get information on customers who had a job loss?"
print(f"Query: {query3}\n")
res3 = search_docs(df_filtered_customers, query3, top_n=2)
print(res3)
print("\n")
print("\n")

query4 = "Filter results based on customers who had a job loss"
print(f"Query: {query4}\n")
res4 = search_docs(df_filtered_customers, query4, top_n=2)
print(res4)
print("\n")
print("\n")
















