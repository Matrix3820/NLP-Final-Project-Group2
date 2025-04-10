import json
import os
import pandas as pd

def format_recipe_for_rag(row):
    label_cols = [
        "diabetes_safe", "hypertension_safe", "heart_disease_safe", "pregnancy_safe",
        "weight_loss", "muscle_gain", "gluten_free_safe", "lactose_free_safe",
        "kidney_disease_safe", "ibs_safe", "cholesterol_friendly", "pcos_friendly",
        "gout_safe", "anemia_support", "thyroid_friendly", "acid_reflux_safe"
    ]
    active_labels = [label for label in label_cols if row[label]]

    try:
        ingredients = eval(row["NER"])
    except Exception:
        ingredients = []
    ingredients_text = ", ".join(ingredients)
    directions_text = "\n".join(eval(row["directions"])) if isinstance(row["directions"], str) else ""

    return {
        "id": f"recipe_{row.name}",
        "title": row["title"],
        "text": f"Ingredients: {ingredients_text}\nInstructions: {directions_text}",
        "labels": active_labels
    }

if __name__ == "__main__":
    CWD = os.getcwd()
    file = 'Data/recipes_data_sample_labelled.csv'
    filepath = os.path.join(CWD, file)
    print(filepath)

    df = pd.read_csv(filepath)

    recipes_for_rag = df.apply(format_recipe_for_rag, axis=1).tolist()

    output_path = "Data/Rag_recipies.json"
    with open(output_path, "w") as f:
        for recipe in recipes_for_rag:
            json.dump(recipe, f)
            f.write("\n")

    print("RAG JSON Created")
