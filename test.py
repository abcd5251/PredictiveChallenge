import csv
import json
import pandas as pd
from utils.similarity_utils import get_embedding, euclidean_distance, manhattan_distance, cosine_similarity, normalize_column

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return {item['github_link']: item['description'] for item in data}

def read_csv(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows

def map_description_to_projects(json_path, csv_path, output_path):
    github_to_description = read_json(json_path)
    csv_rows = read_csv(csv_path)

    output_data = []

    for idx, row in enumerate(csv_rows):
        
        project_a = row['project_a']
        project_b = row['project_b']
        
        description_a = github_to_description.get(project_a, "Not Found")
        description_b = github_to_description.get(project_b, "Not Found")
       
        output_data.append({
            "project_a": project_a,
            "description_a": description_a,
            "project_b": project_b,
            "description_b": description_b
        })

    return output_data

if __name__ == '__main__':
    # Define model
   
    json_file_path = 'new_data.json'  
    csv_file_path = 'test.csv'            
    output_csv_path = 'result.csv'        

    df = pd.read_csv("sample_submission.csv")
    ids = df["id"].tolist()
    preds = df["pred"].tolist()
    output_data = map_description_to_projects(json_file_path, csv_file_path, output_csv_path)

    cos_sim_values = []
    distance_values = []
    mah_distance_values = []
    
    for idx, data in enumerate(output_data):
        print(idx)
        
        embedding_a = get_embedding(data["description_a"])
        embedding_b = get_embedding(data["description_b"])
  
        cos_sim = cosine_similarity(embedding_a, embedding_b)
        distance = euclidean_distance(embedding_a, embedding_b)
        mah_distance = manhattan_distance(embedding_a, embedding_b)
        
        cos_sim_values.append(cos_sim)
        distance_values.append(distance)
        mah_distance_values.append(mah_distance)
        
    # cos_sim_normalized = normalize_column(cos_sim_values)
    # distance_normalized = normalize_column(distance_values)
    # mah_distance_normalized = normalize_column(mah_distance_values)

    with open(output_csv_path, 'w', newline='') as file:
        fieldnames = ['id', 'preds_cos', 'preds_eulidean', 'preds_mah']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, data in enumerate(output_data):
            row = {
                'id': ids[idx],
                'preds_cos': cos_sim_values[idx],
                'preds_eulidean': distance_values[idx],
                'preds_mah': mah_distance_values[idx]
            }

            writer.writerow(row)