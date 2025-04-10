import pandas as pd
import numpy as np

import os

def Label_Data_Rule_Based(row):
    try:
        ingredients = eval(row['NER'].lower())
    except Exception:
        ingredients = []

    ingredient_text = " ".join(ingredients)

    labels = {
        "diabetes_safe": True,
        "hypertension_safe": True,
        "heart_disease_safe": True,
        "pregnancy_safe": True,
        "weight_loss": True,
        "muscle_gain": True,
        "gluten_free_safe": True,
        "lactose_free_safe": True,
        "kidney_disease_safe": True,
        "ibs_safe": True,
        "cholesterol_friendly": True,
        "pcos_friendly": True,
        "gout_safe": True,
        "anemia_support": False,
        "thyroid_friendly": True,
        "acid_reflux_safe": True
    }

    # Diabetes
    if any(k in ingredient_text for k in ['sugar', 'honey', 'corn syrup', 'sweetened', 'molasses']):
        labels["diabetes_safe"] = False

    # Hypertension & Heart disease
    if any(k in ingredient_text for k in ['salt', 'soy sauce', 'bacon', 'processed meat', 'sausage']):
        labels["hypertension_safe"] = False
        labels["heart_disease_safe"] = False

    # Pregnancy
    if any(k in ingredient_text for k in ['raw egg', 'unpasteurized', 'alcohol']):
        labels["pregnancy_safe"] = False

    # Weight loss
    if any(k in ingredient_text for k in ['butter', 'cream', 'fried', 'cheese']):
        labels["weight_loss"] = False

    # Muscle gain
    if any(k in ingredient_text for k in ['chicken', 'beef', 'egg', 'fish', 'tofu']):
        labels["muscle_gain"] = True
    else:
        labels["muscle_gain"] = False

    # Gluten-Free
    if any(k in ingredient_text for k in ['wheat', 'barley', 'rye', 'malt']):
        labels["gluten_free_safe"] = False

    # Lactose-Free
    if any(k in ingredient_text for k in ['milk', 'cheese', 'butter', 'cream']):
        labels["lactose_free_safe"] = False

    # Kidney Disease
    if any(k in ingredient_text for k in ['banana', 'potato', 'tomato', 'cheese', 'nuts']):
        labels["kidney_disease_safe"] = False

    # IBS
    if any(k in ingredient_text for k in ['garlic', 'onion', 'legumes', 'milk', 'wheat']):
        labels["ibs_safe"] = False

    # Cholesterol
    if any(k in ingredient_text for k in ['red meat', 'bacon', 'sausage', 'cream']):
        labels["cholesterol_friendly"] = False

    # PCOS
    if any(k in ingredient_text for k in ['sugar', 'refined flour', 'white bread', 'sweet']):
        labels["pcos_friendly"] = False

    # Gout
    if any(k in ingredient_text for k in ['liver', 'sardines', 'anchovies', 'red meat', 'beer']):
        labels["gout_safe"] = False

    # Anemia
    if any(k in ingredient_text for k in ['spinach', 'beef', 'lentils', 'fortified cereal']):
        labels["anemia_support"] = True

    # Thyroid
    if any(k in ingredient_text for k in ['soy', 'broccoli', 'kale']):
        labels["thyroid_friendly"] = False

    # Acid Reflux
    if any(k in ingredient_text for k in ['tomato', 'citrus', 'chili', 'caffeine', 'chocolate']):
        labels["acid_reflux_safe"] = False

    return labels

if __name__ == "__main__":
    CWD = os.getcwd()
    file = 'Data/recipes_data.csv'
    filepath = os.path.join(CWD, file)
    print(filepath)

    outfile = 'Data/recipes_data_labelled.csv'
    outfile_sample = 'Data/recipes_data_sample_labelled.csv'

    df = pd.read_csv(filepath)
    df = df[['title', 'ingredients','directions','NER']]

    labelled_df_ner = df.copy()
    label_data_ner = labelled_df_ner.apply(Label_Data_Rule_Based, axis=1, result_type='expand')
    labelled_df_ner = pd.concat([labelled_df_ner, label_data_ner], axis=1)

    print("Data Labelled")
    labelled_df_ner.to_csv(outfile, index=False)
    labelled_df_ner.sample(n=200000, random_state=42).to_csv(outfile_sample, index=False)

    print("Files Saved")

