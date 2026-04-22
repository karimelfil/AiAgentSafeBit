import json

# Function to load JSON data from a file
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f)
