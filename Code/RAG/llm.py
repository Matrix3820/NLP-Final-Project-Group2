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

# ========== Load Environment Variables ==========
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "medical-recipes2"

# ========== Load Embedding Model ==========
print(" Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

# ========== Load Pinecone Index ==========
print(" Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ========== Load Model ==========
print(" Loading LLaMA model...")
model_id = "meta-llama/Llama-3.2-3B"
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# model_id = 'tiiuae/falcon-7b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
device = next(model.parameters()).device
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
max_length = 2000
temperature = 0.5

# ========== RAG Function ==========
def generate_medical_recipe_response(query, top_k=5):
    print(f"\n Query: {query}")

    # Embed query
    query_vector = embedding_model.encode(query).tolist()

    # Retrieve from Pinecone
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    if not results['matches']:
        return "No relevant recipes found."

    # Build context
    context = ""
    i=0
    for match in results['matches']:
        i = i+1
        title = match["metadata"].get("title", "Unknown Title")
        ingredients = match["metadata"].get("ingredients", [])
        directions = match["metadata"].get("directions", [])
        context += f"Recipe Name: {title}\n\nIngredients: {'\n'.join(ingredients)}\n\nCooking Instructions: {'\n'.join(directions)}\n\n"

#     # Construct prompt
    prompt = f""" You are a health-conscious culinary assistant.
    ### Context:
    {context}

    ### Instruction:
    For Each Recipe present in the Context:
    * Give the Name of the Recipe followed by :
    * Ingredients required for the recipe, Line Seperated with a number (1,2,3.. and so on)
    * Generate detailed, step-by-step Instructions to prepare this recipe using the listed ingredients.
    * And Lastly explain why this recipe is suitable for the user’s medical conditions and health goals based on the Ingredients used in the recipie.
    
    Respond in a way like you are talking to the user directly.
    ### Response:"""

    prompt = f""" You are a health-conscious culinary assistant.
        ### Context:
        
        Ingredients:
        1. Boneless, skinless chicken breast (4 oz)
        2. Bell pepper (1 large)
        3. Zucchini (1 medium)
        4. Red onion (1 small)
        5. Olive oil
        6. Salt
        7. Black pepper
        8. Garlic powder
        9. Paprika
        10. Lemon juice

        Preparation:
        1. Preheat your grill to medium-high heat.
        2. Cut the chicken breast into bite-sized pieces.
        3. Cut the bell pepper, zucchini, and red onion into bite-sized pieces.
        4. Place the chicken, bell pepper, zucchini, and red onion in a bowl.
        5. Drizzle the olive oil over the vegetables and chicken.
        6. Season the vegetables and chicken with salt, black pepper, garlic powder, and paprika.
        7. Toss the vegetables and chicken to coat them evenly with the seasoning.
        8. Squeeze lemon juice over the vegetables and chicken.
        9. Place the vegetables and chicken on the grill.
        10. Grill the vegetables and chicken for 5-7 minutes on each side, or until the chicken is cooked through and the vegetables are tender.
        11. Serve the grilled chicken with roasted vegetables on the side.


        ### Instruction:
        For Each Recipe present in the Context:
        * Give the Name of the Recipe
        * Ingredients required for the recipe, Line Seperated with a number (1,2,3.. and so on)
        * Generate detailed, step-by-step Instructions to prepare this recipe using the listed ingredients.
        * Educate the user on why this recipie is MEDICALLY GOOD for them based on the ingredients used in it and cooking instructions(if beneficial)

        ### Response:"""

    prompt = f""" You are a health-conscious culinary assistant.

            \nUser Query: {query}\n
            
            ### Context:
            {context}\n
            
            ### Instruction:
            For Each Recipe present in the Context:
            * Give the Name of the Recipe
            * Ingredients required for the recipe, Line Seperated with a number (1,2,3.. and so on)
            * Generate detailed, step-by-step Instructions to prepare this recipe using the listed ingredients.
            * Educate the user on why this recipie is MEDICALLY GOOD for them based on the ingredients used in it and cooking instructions(if beneficial)

            Example:

            1.Recipie Name
            
            2.Ingredients:
                a. Ingredient 1 Name
                b. Ingredient 2 Name
                c. Ingredient 3 Name
            
            3.Cooking Instructions:
                a. Step1
                b. Step2
            
            4.Explanation why the recipe is good"""

    # # Generate response
    # output = llm(prompt, max_new_tokens=1000, do_sample=True)[0]["generated_text"]
    # return output[len(prompt):].strip()

    # prompt =(
    #     "<System Prompt>\n"
    #     "You are a health-conscious culinary assistant.\n\n"
    #
    #     "For each Recipe in the Context provide the following information in the order provided:\n"
    #     "- The Name of the Recipe\n"
    #     "- The Ingredients used - Line by Line\n"
    #     "- The Cooking Instructions - Step by Step\n"
    #     "- The Medically factual response as to why this recipe is good for the user based on the Ingredients used\n\n"
    #
    #
    #     "</System Prompt>\n\n"
    #     "Query: {question}\n\n"
    #     "Context: {context}\n\n"
    #     "Your Response:".format(question=query,context=context)
    # )

    # prompt = (
    #     "<|system|>\nYou are a health-conscious culinary assistant.\n"
    #     "For each recipe provided, Generate the following sections:\n"
    #     "1. Recipe Title\n"
    #     "2. Ingredients (one per line)\n"
    #     "3. Instructions (step by step)\n"
    #     "4. Medical Explanation: Explain why this recipe is beneficial, based on the ingredients used and the user’s dietary needs.\n"
    #     "</|system|>\n\n"
    #
    #     "<|user|>\n{question}\n\nContext:\n{context}\n</|user|>\n\n"
    #     "<|assistant|>\n"
    # ).format(question=query, context=context)

    # prompt = (
    #     "You are a health-conscious culinary assistant.\n\n"
    #     "Given the following recipe context, for each recipe, extract and return:\n"
    #     "1. Recipe Title\n"
    #     "2. Ingredients (one per line)\n"
    #     "3. Cooking Instructions (step by step)\n"
    #     "4. Medical Explanation: Explain why this recipe is beneficial, based on the ingredients used and the user's health concerns.\n\n"
    #     "Question: {question}\n\n"
    #     "Context:\n{context}\n\n"
    #     "Response:\n"
    # ).format(question=query, context=context)

    prompt = f"""<|system|>
    You are a health-conscious culinary assistant.
    </|system|>

    <|user|>
    User query: {query}

    Context:
    {context}

    Instructions:
    For each recipe found in the context:
    1. Print the recipe name.
    2. List the ingredients as a numbered list.
    3. Write clear, step-by-step cooking instructions.
    4. Explain why the recipe is medically beneficial based on its ingredients and cooking method.
    </|user|>

    <|assistant|>
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        # max_length=max_length,
        max_new_tokens=500,
        # temperature=temperature,
        repetition_penalty=1.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Remove the input text from the generated output
    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)

    return response

# ========== Non-RAG Function ==========
def generate_llm_only_response(query):
    print(f"\n Query (LLM only): {query}")

    prompt = f"""You are a health-conscious culinary assistant. Given the user's request

User Query: {query}

### Instruction:
* Give the Name of the Recipe
* Ingredients required for the recipe, Line Seperated with a number (1,2,3.. and so on)
* Generate detailed, step-by-step Instructions to prepare this recipe using the listed ingredients.
* Educate the user on why this recipie is MEDICALLY GOOD for them based on the ingredients used in it and cooking instructions(if beneficial)

Example:

Recipie Name

Ingredients:
1. Ingredient 1 Name
2. Ingredient 2 Name
3. Ingredient 3 Name

Directions:
1. Step1
2. Step2

Explanation why the recipie is good
Response:"""

    # output = llm(prompt, max_new_tokens=1000, do_sample=True)[0]["generated_text"]
    # return output[len(prompt):].strip()
    prompt = (
        "<System Prompt>\n"
        "You are a health-conscious culinary assistant.\n"
        "Read the User's query and provide a clear, and medically factual response.\n\n"
        "Your answer should include:\n"
        "- The Name of the Recipe\n"
        "- The Ingredients used\n"
        "- The Cooking Directions\n"
        "- The Medically factual response as to why this recipe is good for the user\n\n"

        "</System Prompt>\n\n"
        "Query: {question}\n\n"
        "Your Response:".format(question=query)
    )

    prompt = (
        "<|system|>\nYou are a health-conscious culinary assistant.\n"
        " Read the User's query and provide a clear, and medically factual response. Return the following sections:\n"
        "1. Recipe Title\n"
        "2. Ingredients (one per line)\n"
        "3. Instructions (step by step)\n"
        "4. Medical Explanation: Explain why this recipe is beneficial, based on the ingredients used and the user’s dietary needs.\n"
        "</|system|>\n\n"

        "<|user|>\n{question}</|user|>\n\n"
        "<|assistant|>\n"
    ).format(question=query)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        # temperature=temperature,
        # repetition_penalty=1.2,
        do_sample=False,
        # pad_token_id=tokenizer.eos_token_id,
        # eos_token_id=tokenizer.eos_token_id
    )
    # Remove the input text from the generated output
    generated_tokens = outputs[0]
    input_length = inputs["input_ids"].shape[1]
    output_final = generated_tokens[input_length:]
    response = tokenizer.decode(output_final, skip_special_tokens=True)

    return response

# ========== Main ==========
if __name__ == "__main__":
    user_query = "I need a dinner that is safe for people with diabetes to help with building muscle"

    print("\n Running with RAG...")
    rag_response = generate_medical_recipe_response(user_query)
    print(f"\n RAG Response:\n{rag_response}")

    print("\n Running without RAG (LLM only)...")
    llm_response = generate_llm_only_response(user_query)
    print(f"\n LLM-Only Response:\n{llm_response}")
