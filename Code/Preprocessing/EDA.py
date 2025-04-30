import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the labeled dataset
df = pd.read_csv('Data/recipes_data_labelled.csv')

# Clean and standardize ingredients
def extract_ingredients(ner_str):
    try:
        return [i.strip().lower() for i in ast.literal_eval(ner_str)]
    except:
        return []

# Flatten all ingredients into a single list
all_ingredients = df['NER'].dropna().apply(extract_ingredients).explode()

# Count frequencies
ingredient_counts = Counter(all_ingredients)

# Convert to DataFrame
ingredient_freq_df = pd.DataFrame(ingredient_counts.items(), columns=['ingredient', 'frequency'])
ingredient_freq_df = ingredient_freq_df.sort_values(by='frequency', ascending=False)

# Save to CSV
ingredient_freq_df.to_csv('Data/ingredient_frequencies.csv', index=False)

# Plot top 20 ingredients
plt.figure(figsize=(12, 6))
sns.barplot(data=ingredient_freq_df.head(20), x='frequency', y='ingredient', palette='viridis')
plt.title('Top 20 Most Common Ingredients')
plt.xlabel('Frequency')
plt.ylabel('Ingredient')
plt.tight_layout()
# plt.show()
plt.savefig('Data/ingredient_frequencies_20.png',dpi=300)
