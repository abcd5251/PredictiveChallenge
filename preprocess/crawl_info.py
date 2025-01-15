import os
from github import Github
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
df = pd.read_csv("data1.csv")
artifacts_namespace = df["artifact_namespace"].tolist()
artifacts_name = df["artifact_name"].tolist()

g = Github(os.getenv("GITHUB_TOKEN"))

repo = g.get_repo(f"{artifacts_namespace[0]}/{artifacts_name[0]}")

print(f"repo name: {repo.name}")
print(f"description: {repo.description}")
print(f"number of star: {repo.stargazers_count}")
print(f"number of branch: {repo.get_branches().totalCount}")

contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        print(f"file path: {file_content.path}")
