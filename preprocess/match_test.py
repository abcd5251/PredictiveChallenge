import csv
import json

# Load data from a CSV file
def load_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

# Load data from a JSON file
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Find the description for a given GitHub link in the descriptions
def find_description(github_link, descriptions):
    for entry in descriptions:
        if entry["github_link"] == github_link:
            return entry["description"]
    return None

# Find repository information for a given GitHub link in repo data
def find_repo_info(github_link, repo_data):
    return repo_data.get(github_link, {}).get("information", {})

# Main function to process data
def process_data(csv_data, json_data, repo_data):
    enriched_data = []

    for row in csv_data:
        enriched_row = row.copy()

        # Process project_a
        project_a_desc = find_description(row["project_a"], json_data)
        if project_a_desc:
            for key, value in project_a_desc.items():
                enriched_row[f"project_a_{key}"] = value

        project_a_repo_info = find_repo_info(row["project_a"], repo_data)
        if project_a_repo_info:
            for key, value in project_a_repo_info.items():
                enriched_row[f"project_a_repo_{key}"] = value

        # Process project_b
        project_b_desc = find_description(row["project_b"], json_data)
        if project_b_desc:
            for key, value in project_b_desc.items():
                enriched_row[f"project_b_{key}"] = value

        project_b_repo_info = find_repo_info(row["project_b"], repo_data)
        if project_b_repo_info:
            for key, value in project_b_repo_info.items():
                enriched_row[f"project_b_repo_{key}"] = value

        enriched_data.append(enriched_row)

    return enriched_data

# Save enriched data to a new CSV file
def save_to_csv(file_path, data):
    if not data:
        return

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Paths to files
test_csv_path = "test.csv"
data_json_path = "new_new_data.json"
repo_data_path = "merged_data.json"
enriched_csv_path = "test_data.csv"

# Load files
csv_data = load_csv(test_csv_path)
json_data = load_json(data_json_path)
repo_data = load_json(repo_data_path)

# Process data
enriched_data = process_data(csv_data, json_data, repo_data)

# Save enriched data
save_to_csv(enriched_csv_path, enriched_data)

print(f"Enriched data has been saved to {enriched_csv_path}.")
