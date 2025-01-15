import os
import asyncio
import json
from playwright.async_api import async_playwright
import pandas as pd
import uuid
async def capture_github_screenshot(url, output_dir='images'):
    output_file=f'{uuid.uuid4()}.jpg'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    async with async_playwright() as p:

        browser = await p.chromium.launch()

        page = await browser.new_page()
        
        await page.goto(url)
    
        await page.wait_for_load_state('load')

        await page.screenshot(path=output_path, full_page=True)

        await browser.close()

    print(f'Save: {output_path}')
    return output_path

def append_to_json_file(image_path, project, json_file='data.json'):
    new_entry = {"image_path": image_path, "github_link": project}
    print(f"Starting to append {new_entry}!")
    if not os.path.exists(json_file):
        with open(json_file, 'w') as file:
            json.dump([], file)

    with open(json_file, 'r') as file:
        data = json.load(file)

    data.append(new_entry)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Successfully appended {new_entry}!")
    

if __name__ == '__main__':
    df = pd.read_csv("data1.csv")
    project_a = list(df["project_a"])
    project_b = list(df["project_b"])
    projects = project_a + project_b
    github_links = set()
    

    for project in projects:
        if project not in github_links:
            github_links.add(project)
            image_path = asyncio.run(capture_github_screenshot(project))
            append_to_json_file(image_path, project)

    print("Done")