import os
import json

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
login(token="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")

# %%

# ========== Load Environment Variables ==========
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
RECIPE_INDEX_NAME = os.getenv("RECIPE_INDEX_NAME")
FACTS_INDEX_NAME = os.getenv("FACTS_INDEX_NAME")

# ========== Load Embedding Model ==========
print(" Loading embedding model...")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ========== Load Pinecone Indexes ==========
print(" Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
recipe_index = pc.Index(RECIPE_INDEX_NAME)
ingredient_index = pc.Index(FACTS_INDEX_NAME)

# ========== Load Model ==========
print(" Loading LLaMA model...")
# model_id = "meta-llama/Llama-3.2-3B"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
device = next(model.parameters()).device
max_new_tokens = 1500
maxlen = 6500

def format_dual_rag_prompt(query, recipes):
    """
        Constructs a structured prompt for the RAG system using a user query, relevant recipes,
        and associated ingredient-level medical facts.

        Parameters
        ----------
        query : str
            The user-provided dietary or health-related query.
        recipes : list of dict
            A list of dictionaries, each representing a recipe with keys like
            'title', 'ingredients', 'directions', and 'matched_facts'.

        Returns
        -------
        str
            A formatted prompt string that incorporates the user's query,
            recipe details, and medical knowledge for LLM-based generation.
        """

    prompt = f"""### Instruction:
You are a health-conscious culinary assistant.

Given a user query, a set of recipes, and ingredient-level medical facts, answer in a helpful and medically-informed way.

### Input:
User Query: {query}

"""
#     prompt = f"""### Instruction:
# You are a health-conscious culinary assistant.
#
# Given a user query, a set of recipes, and ingredient-level medical facts, Generate a Medically Aware Recipe.
#
# ### Input:
# User Query: {query}
#
# """

    for recipe in recipes:
        title = recipe.get("title", "Unknown Title")
        ingredients = recipe.get("ingredients", [])
        directions = recipe.get("directions", [])
        prompt += f"\nRecipe Name: {title}\n"
        prompt += "Ingredients:\n" + "\n".join(f"- {ing}" for ing in ingredients) + "\n"
        prompt += "Instructions:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(directions)) + "\n"

        prompt += "\nIngredient-Level Medical Facts:\n"
        # for recipe in recipes:
        for fact in recipe.get("matched_facts", []):
            prompt += f"- {fact}\n"

    prompt += """

### Response:
Please suggest a recipe that is medically suitable based on the user's request. Do not repeat Recipes

List:
1. Recipe Name
2. Ingredients (numbered list)
3. Cooking Instructions (numbered list)
4. Medical Explanation (ingredient-wise health knowledge)

"""
    return prompt.strip()


def generate_medical_recipe_response(query, top_k=3):
    """
        Generates a medically tailored recipe using a dual RAG (Retrieval-Augmented Generation) pipeline.

        Parameters
        ----------
        query : str
            A natural language user query related to diet or health goals.
        top_k : int, optional
            Number of top recipes to retrieve from the Pinecone recipe index (default is 3).

        Returns
        -------
        str
            A generated response that includes a medically suitable recipe, ingredients,
            cooking steps, and ingredient-wise health benefits.

        Notes
        -----
        - Embeds the user query to retrieve relevant recipes and ingredient facts.
        - Constructs a prompt that combines retrieved content and generates a final response using Mistral-7B.
        """
    # print(f"\n Query: {query}")

    # Step 1: Embed query and query recipe index
    query_vector = embedding_model.encode(query).tolist()
    recipe_results = recipe_index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    if not recipe_results['matches']:
        return "No relevant recipes found."

    # Step 2: Parse recipes and extract ingredients
    recipes = []
    all_ingredients = set()

    for match in recipe_results['matches']:
        metadata = match['metadata']
        ingredients = metadata.get("ingredients", [])
        recipes.append({
            "title": metadata.get("title", "Unknown Title"),
            "ingredients": ingredients,
            "directions": metadata.get("instructions", [])
        })
        all_ingredients.update(ing.lower().strip() for ing in ingredients)

    # Step 3: Query ingredient index for multiple medical facts
    ingredient_fact_map = {}
    for ingredient in all_ingredients:
        ing_vector = embedding_model.encode(ingredient).tolist()
        ing_results = ingredient_index.query(vector=ing_vector, top_k=3, include_metadata=True)
        for match in ing_results['matches']:
            fact = match['metadata'].get("fact")
            title = match['metadata'].get("title", "").lower().strip()
            if title and fact:
                if title not in ingredient_fact_map:
                    ingredient_fact_map[title] = []
                if fact not in ingredient_fact_map[title]:
                    ingredient_fact_map[title].append(fact)

    # Step 4: Attach all matching facts to each recipe
    for recipe in recipes:
        matched_facts = []
        for ing in recipe['ingredients']:
            ing_l = ing.lower().strip()
            if ing_l in ingredient_fact_map:
                for fact in ingredient_fact_map[ing_l]:
                    matched_facts.append(f"{ing}: {fact}")
        recipe["matched_facts"] = matched_facts

    # Step 5: Build the final prompt using Mistral-style instructions
    prompt = format_dual_rag_prompt(query, recipes)
    # print(f"[DEBUG] Prompt Length (chars): {len(prompt)}")
    # print(f"[DEBUG] Prompt Preview:\n{prompt}")

    # Step 6: Tokenize and generate response using Mistral
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=maxlen).to(device)
    # print(f"[DEBUG] Tokenized input shape: {inputs['input_ids'].shape}")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=0.12,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,

        eos_token_id=tokenizer.eos_token_id
    )

    # outputs = model.generate(
    #     input_ids=inputs["input_ids"],
    #     attention_mask=inputs["attention_mask"],
    #     max_new_tokens=max_new_tokens,
    #     # temperature=0.7,
    #     # top_p=0.95,
    #     repetition_penalty=0.9,
    #     do_sample=False,
    #     pad_token_id=tokenizer.eos_token_id,
    #
    #     eos_token_id=tokenizer.eos_token_id
    # )
    # Remove prompt tokens from output
    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)

    return response


def format_llm_only_prompt(query):
    """
        Formats a minimal prompt using only the user query, without any external retrieval,
        for baseline comparison against the RAG-enhanced system.

        Parameters
        ----------
        query : str
            The user's recipe or diet-related question.

        Returns
        -------
        str
            A simple prompt designed to instruct the LLM to generate a medically informed recipe.
        """
    return f"""### Instruction:
You are a health-conscious culinary assistant.

### Input:
{query}

### Response:
Please suggest a recipe that is medically suitable based on the user's request.

List:
1. Recipe Name
2. Ingredients (numbered list)
3. Cooking Instructions (numbered list)
4. Medical Explanation (ingredient-wise health knowledge)
"""

def generate_llm_only_response(query):
    """
        Generates a recipe using only the language model, without any external retrieval or ingredient facts.

        Parameters
        ----------
        query : str
            A natural language query related to diet, ingredients, or medical constraints.

        Returns
        -------
        str
            The response generated by the LLM (e.g., Mistral-7B), based solely on the prompt.

        Notes
        -----
        - Serves as a baseline comparison to evaluate the benefit of the RAG pipeline.
        """

    prompt = format_llm_only_prompt(query)
    # print(f"[DEBUG] LLM-Only Prompt:\n{prompt[:1000]}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=maxlen).to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    output_final = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)
    return response


# ========== Main ==========
if __name__ == "__main__":
    # user_query = " I need a dinner that is safe for people with diabetes to help with building muscle"
    # user_query = " Seafood recipe for building muscle"
    # user_query = "I need a recipe for a diabetic diet"
    # user_query = "Generate a Thyroid friendly recipe for weight loss" # Both bad
    # user_query = "I have anemia. Generate a recipe for building muscle" #---------#
    # user_query = "Generate a recipe for someone with heart disease and cardiovascular issues" #----------best
    # user_query = "Generate a Recipe with Beef, Brocolli and Garlic" #--

    user_query = input("Enter your Query: Eg - Generate a recipe for someone with heart disease and cardiovascular issues")
    print("You entered:", user_query)

    print("\n Running with RAG...")
    rag_response = generate_medical_recipe_response(user_query,)
    print(f"\n RAG Response:\n{rag_response}")

    print("\n Running without RAG (LLM only)...")
    llm_response = generate_llm_only_response(user_query)
    print(f"\n LLM-Only Response:\n{llm_response}")
