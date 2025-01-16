import csv
import json
import time
import pandas as pd
from models.model import OpenAIModel
from prompts.get_info_prompt import crawl_analyze_planner
import asyncio
from crawl4ai import *


async def crawl_information(github_link):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=github_link,
        )
    return result.markdown


async def main():
    openai_model_instance = OpenAIModel(
        system_prompt=crawl_analyze_planner,
        temperature=0
    )
    dataset_df = pd.read_csv("dataset.csv")
    test_df = pd.read_csv("test.csv")
    target_repos = dataset_df["project_a"].tolist() + dataset_df["project_b"].tolist() + test_df["project_a"].tolist() + test_df["project_b"].tolist()
    repo_urls = list(set(target_repos))
    print(len(repo_urls))
    total_list = []

    for github_link in repo_urls:
        information = await crawl_information(github_link)  
        prompt = f"###INFORMATION: {information}\n###JSON_REPLY:\n"
        output_text, input_token, output_token = openai_model_instance.generate_json(prompt)
        output_text = output_text.replace("```", "").replace("json", "")
        item = {"github_link": github_link, "description": json.loads(output_text)}
        print("item", item)
        total_list.append(item)
        with open("new_new_data.json", "w") as file:
            json.dump(total_list, file)
        await asyncio.sleep(0.5)  

if __name__ == '__main__':
    asyncio.run(main())  
