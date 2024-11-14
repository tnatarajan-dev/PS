import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# Ignore all warnings
import warnings
warnings.simplefilter('ignore')


# Setting up the deployment name
model_name = "text-embedding-ada-002"

# The base URL for your Azure OpenAI resource
#"https://genai-openai-profitsentinel.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?"
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
df = pd.read_csv(os.path.join(os.getcwd(),'eval_test_data.txt'), sep='|')
# Strip call whitespace from dataframe
df = df.applymap(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
print(df)

# Interested columns from the spreadsheet
df_customers = df[['customer', 'riskcategory', 'analysis']]
print(df_customers)

client = AzureOpenAI(
    api_key = openai_api_key,
    api_version = openai_api_version,
    azure_endpoint = api_url
)
print("\n")
print("\n")

df_filtered_customers = df_customers[df['riskcategory'] == 'Very High Risk']
df_filtered_customers['ada_v2'] = df_filtered_customers['analysis'].apply(lambda x : generate_embeddings (x, model = model_name))
print(df_filtered_customers)
print("\n")
print("\n")

df_reason = pd.read_csv(os.path.join(os.getcwd(),'seg_test_data.txt'), sep='|')
# Strip call whitespace from dataframe
df_reason = df_reason.applymap(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
print(df_reason)
print("\n")
print("\n")

df_reason = pd.merge(df_filtered_customers, df_reason, on='customer', how='inner')
print(df_reason)
print("\n")
print("\n")

columns = ["financial distress", "excessive debt", "life changes" ]
dict = {
    columns[0] : generate_embeddings(columns[0], model_name),
    columns[1] : generate_embeddings(columns[1], model_name),
    columns[2] : generate_embeddings(columns[2], model_name)
}
#print(dict)

for index, row in df_reason.iterrows():

    embedding = generate_embeddings(row['reason'], model_name)
    distress = cosine_similarity(dict[columns[0]], embedding)
    debt = cosine_similarity(dict[columns[1]], embedding)
    changes = cosine_similarity(dict[columns[2]], embedding)

    print(f"{index} similarity for distress: {distress}")
    print(f"{index} similarity for debt: {debt}")
    print(f"{index} similarity for changes: {changes}")

    if(distress > debt and distress > changes):
        df_reason.at[index, 'segment'] = columns[0]
    if(debt > distress and debt > changes):
        df_reason.at[index, 'segment'] = columns[1]
    if(changes > distress and changes > debt):
        df_reason.at[index, 'segment'] = columns[2]


#print(df_reason)
headers = ["customer", "riskcategory", "analysis", "reason", "segment"]
df_reason.to_csv("Segment_Data.csv", sep='|', columns=headers)

# Bar Chart
'''
value_counts = df_reason['segment'].value_counts()
print(value_counts)
print(type(value_counts))
value_counts.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.title('Value Counts of Categories')
plt.show()
'''

pie = df_reason.groupby('segment').size().to_dict()

# Step 1: Extract labels and values from the dictionary
labels = list(pie.keys())
values = list(pie.values())

# Step 2: Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('High Risk Customers Category')
plt.savefig('pie_chart.png')


cutoff_date = datetime.strptime("06/01/2024", '%m/%d/%Y').date()

full_data = pd.read_csv("Testing_Data_NEW.csv", sep=",")
#full_data['Snapshot Month'] = pd.to_datetime(full_data['Snapshot Month']).dt.date
full_data['Snapshot Month'] = pd.to_datetime(full_data['Snapshot Month'],  format="%m/%d/%Y").dt.date
full_data = full_data[full_data['Snapshot Month'] > cutoff_date]
print(full_data)


df_combined = pd.merge(df_reason, full_data, left_on='customer', right_on='Customer ID', how='left')
print(df_combined)


#df_sorted = df_combined.sort_values(by='Snapshot Month')
df_combined['Snapshot Month'] = pd.to_datetime(df_combined['Snapshot Month'], format='%b')
print(df_combined)


# Groupby by Snapshot month
line_chart = df_combined.groupby(['Snapshot Month']).agg({
    'Utilization': 'mean',
    'FICO': 'mean',
    'Total_Debt': 'mean',
    'Cumulative Profit': 'mean',
}).reset_index()
print(line_chart)
print(type(line_chart))

# Define data values
line_chart.plot(x='Snapshot Month', y='Utilization', kind='line')
plt.xlabel('Month')
plt.ylabel('Utilization')
plt.title('Utilization Curve')
plt.savefig('utilization.png')

# Define data values
line_chart.plot(x='Snapshot Month', y='FICO', kind='line')
plt.xlabel('Month')
plt.ylabel('FICO')
plt.title('FICO Curve')
plt.savefig('fico.png')


# Define data values
line_chart.plot(x='Snapshot Month', y='Total_Debt', kind='line')
plt.xlabel('Month')
plt.ylabel('Total_Debt')
plt.title('Debt Curve')
plt.savefig('total_debt.png')


# Define data values
line_chart.plot(x='Snapshot Month', y='Cumulative Profit', kind='line')
plt.xlabel('Month')
plt.ylabel('Cumulative Profit')
plt.title('Cumulative Profit Curve')
plt.savefig('cumulative_profit.png')


'''
num_segments = 3
segment_size = len(df_reason) // num_segments
segments = [df_reason[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]

print(segments)

# Step 3: Calculate magnitude or sum for each segment
# We’ll use the sum here for simplicity
segment_sums = [np.sum(segment) for segment in segments]

print(segment_sums)

# Step 4: Plot a pie chart
labels = [f'Segment {i+1}' for i in range(num_segments)]
plt.figure(figsize=(8, 8))
plt.pie(segment_sums, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Segmented Embedding Vector')
plt.show()
'''