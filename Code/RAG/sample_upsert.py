import json
import os

from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ====== CONFIGURATION ======
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
RECIPE_INDEX_NAME = os.getenv("RECIPE_INDEX_NAME")
FACTS_INDEX_NAME = os.getenv("FACTS_INDEX_NAME")
# INDEX_NAME = "medically-aware-recipes"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
DIMENSION = int(os.getenv("DIMENSION"))
BATCH_SIZE = 64



# ====== Load JSONL Data ======
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# ====== Batch Embedding and Upsert ======
def upsert_to_pinecone(data,_index,model):
    texts = [item["text"] for item in data]
    ids = [item.get("id") or item.get("title") or str(hash(item['text'])) for item in data]
    metas = [item.get("metadata", {}) for item in data]

    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Ingesting"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_metas = metas[i:i+BATCH_SIZE]

        embeddings = model.encode(batch_texts).tolist()
        to_upsert = list(zip(batch_ids, embeddings, batch_metas))

        _index.upsert(vectors=to_upsert)

    print("✅ Recipe Ingestion complete using SentenceTransformers!")

if __name__ == "__main__":
    recipe_data = load_jsonl("Data/Rag_Recipies.jsonl")
    facts_data = load_jsonl("Data/Rag_Ingredients.jsonl")

    # ====== Load SentenceTransformer Model ======
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ====== Initialize Pinecone ======
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if RECIPE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=RECIPE_INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    if FACTS_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=FACTS_INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    recipe_index = pc.Index(RECIPE_INDEX_NAME)
    facts_index = pc.Index(FACTS_INDEX_NAME)

    upsert_to_pinecone(recipe_data, recipe_index, model)
    upsert_to_pinecone(facts_data, facts_index, model)