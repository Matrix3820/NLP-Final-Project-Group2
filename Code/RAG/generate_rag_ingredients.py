import os
import json
import pandas as pd

def format_ingredients_for_rag(row):
    return {
        "id": f"ingredient_{row.name}",
        "text": f"{row['ingredient']} - {row['Fact']}",
        "metadata": {
            "title": row['ingredient'],
            "fact": row['Fact']
        }
    }

if __name__ == "__main__":
    CWD = os.getcwd()
    file = os.path.join(CWD, 'Data/Ingredients.jsonl')

    # Load your JSONL file
    df = pd.read_json(file, lines=True)

    # Format each row for RAG
    formatted_data = [format_ingredients_for_rag(row) for _, row in df.iterrows()]

    # Write output
    output_path = os.path.join(CWD, "Data/Rag_Ingredients.jsonl")
    with open(output_path, "w") as f:
        for item in formatted_data:
            json.dump(item, f)
            f.write("\n")

    print("RAG Ingredients JSON Created")
