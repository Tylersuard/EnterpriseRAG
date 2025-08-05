
import csv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os
from dotenv import load_dotenv
load_dotenv()

client = SearchClient(
    endpoint="https://your_ai_search.search.windows.net",
    index_name="medical_records",
    credential=AzureKeyCredential(os.environ.get("AI_SEARCH_KEY")))
with open("medical_records.csv", newline='', encoding='utf-8') as f:
    docs = list(csv.DictReader(f))
    import ast
    for d in docs:
        d['allowed_roles'] = ast.literal_eval(d['allowed_roles'])
        d['allowed_fields'] = str(d['allowed_fields'])
    print(client.upload_documents(documents=docs))
