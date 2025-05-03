import os
import json
import pandas as pd

def format_ingredients_for_rag(row):
    """
        Formats an ingredient row into a structured dictionary suitable for use in
        a Retrieval-Augmented Generation (RAG) system.

        This function converts each row of the ingredients dataset into a JSON-compatible
        object containing an identifier, display text (combining the ingredient name and fact),
        and metadata for downstream retrieval tasks.

        Parameters
        ----------
        row : pd.Series
            A row from a DataFrame with at least the following columns:
            - 'ingredient': The name of the ingredient
            - 'Fact': A brief medical or nutritional fact about the ingredient

        Returns
        -------
        dict
            A dictionary with the following structure:
            {
                "id": str,                # Unique identifier (e.g., "ingredient_5")
                "text": str,              # Combined ingredient and fact string
                "metadata": {
                    "title": str,         # Ingredient name
                    "fact": str           # Corresponding medical/nutritional fact
                }
            }

        Notes
        -----
        - Assumes the DataFrame index or `row.name` is unique per row for ID generation.
        - Intended for JSONL export to be used in vector databases or RAG pipelines.
        """
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
