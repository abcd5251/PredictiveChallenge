import csv
import json
import time
import pandas as pd
from models.model import OpenAIModel
from prompts.analyze_prompt import analyze_planner


if __name__ == '__main__':
    openai_model_instance = OpenAIModel(
            system_prompt=analyze_planner,
            temperature=0
        )
    with open("data.json", "r") as file:
        data = json.load(file)
    for item in data:
        print(item)
        image_path = item["image_path"]
        image_paths = [image_path]
    
    
        output_text, input_token, output_token = openai_model_instance.process_image(image_paths)
        print(output_text)
        output_text = output_text.replace("```","").replace("json","")
        item["description"] = json.loads(output_text)
        print("itm", item)   
        with open("new_new_data.json", "w") as file:
            json.dump(data, file)