from google.cloud import bigquery
from collections import Counter
import json
import numpy as np
import os
import pandas as pd
import re

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './oso-credentials.json'
GCP_PROJECT = 'oso-bigquery'
client = bigquery.Client(GCP_PROJECT)

def check_weight(g):
    return sum(
        data['weight']
        for _, _, data in g.edges(data=True)
    )

def stringify_array(arr):
    return "'" + "','".join(arr) + "'"

df = pd.read_csv("test.csv")
target_repos = list(df["project_b"])
owners = set()
for target in target_repos:
    match = re.match(r"https://github\.com/([^/]+)", target)
    owner = match.group(1)
    owners.add(owner)
    
results = client.query(f"""
    select distinct * except(project_id, artifact_source)
    from `oso.int_repo_metrics_by_project`
    where artifact_namespace in ({stringify_array(owners)})
""")
df_repo_metrics = results.to_dataframe()
df_repo_metrics['github_url'] = df_repo_metrics.apply(
    lambda x: f"https://github.com/{x['artifact_namespace']}/{x['artifact_name']}", axis=1
)
df_repo_metrics.to_csv('./repo_metrics.csv')



