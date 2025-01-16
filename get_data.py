import os
import json
import time
import pandas as pd
from github import Github
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

access_token = os.getenv("GITHUB_TOKEN")
g = Github(access_token)

def get_repo_info(repo_url):
    parts = repo_url.rstrip('/').split('/')
    owner, repo_name = parts[-2], parts[-1]

    
    repo = g.get_repo(f"{owner}/{repo_name}")

    repo_info = {
        "is_fork": repo.fork,
        "fork_count": repo.forks_count,
        "star_count": repo.stargazers_count,
        "watcher_count": repo.subscribers_count,
        "language": repo.language,
        "license_spdx_id": repo.license.spdx_id if repo.license else None,
        "created_at": repo.created_at,
        "updated_at": repo.updated_at,
    }

    repo_info["exist_days"] = (repo.updated_at - repo.created_at).days

    commits = list(repo.get_commits())
    if commits:
        repo_info["first_commit_time"] = commits[-1].commit.author.date
        repo_info["last_commit_time"] = commits[0].commit.author.date
        repo_info["commit_count"] = len(commits)

        repo_info["work_days"] = (repo_info["last_commit_time"] - repo_info["first_commit_time"]).days

        commit_dates = {commit.commit.author.date.date() for commit in commits}
        repo_info["days_with_commits_count"] = len(commit_dates)
    else:
        repo_info["first_commit_time"] = None
        repo_info["last_commit_time"] = None
        repo_info["commit_count"] = 0
        repo_info["work_days"] = 0
        repo_info["days_with_commits_count"] = 0

    contributors = list(repo.get_contributors())
    repo_info["contributors_to_repo_count"] = len(contributors)

    return repo_info

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == '__main__':
    
    dataset_df = pd.read_csv("dataset.csv")
    test_df = pd.read_csv("test.csv")
    target_repos = dataset_df["project_a"].tolist() + dataset_df["project_b"].tolist() + test_df["project_a"].tolist() + test_df["project_b"].tolist()
    repo_urls = list(set(target_repos))
    print(len(repo_urls))
    
    # Read json
    json_path = "new_data.json"
    with open(json_path, 'r') as file:
        json_list = json.load(file)
    
    all_repo_info = {}

    with open("past_repo_info.json", "r") as file:
        data_past_repo = json.load(file)
    
    for repo_url in repo_urls:
        if repo_url in data_past_repo.keys():
            print(repo_url, "Has already exist!")
            continue
    
        print(repo_url, "start")
        info = get_repo_info(repo_url)

        # Find corresponding data in json_list by github_link
        matched_data = next((item for item in json_list if item["github_link"] == repo_url), None)
        if matched_data:
            info["image_path"] = matched_data["image_path"]
            info["description"] = matched_data["description"]
        else:
            info["image_path"] = None
            info["description"] = None

        all_repo_info[repo_url] = {"information": info}
        print(repo_url, "end")
        time.sleep(0.5)
        # Write updated data to repo_info.json
        with open('repo_info.json', 'w', encoding='utf-8') as f:
            json.dump(all_repo_info, f, ensure_ascii=False, indent=4, default=json_serial)

    # Print out the results
    # for repo_url, data in all_repo_info.items():
    #     print(f"Repository: {repo_url}")
    #     for key, value in data["information"].items():
    #         print(f"  {key}: {value}")
