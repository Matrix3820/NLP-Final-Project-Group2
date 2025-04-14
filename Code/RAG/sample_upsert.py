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
INDEX_NAME = "medically-aware-recipes"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions
BATCH_SIZE = 64

# ====== Initialize Pinecone ======
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-recipes2"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(index_name)

# ====== Load SentenceTransformer Model ======
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ====== Load JSONL Data ======
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

data = load_jsonl("Data/Rag_recipies.jsonl")

# ====== Batch Embedding and Upsert ======
texts = [item["text"] for item in data]
ids = [item.get("id") or item.get("title") or str(hash(item['text'])) for item in data]
metas = [item.get("metadata", {}) for item in data]

for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Ingesting"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_metas = metas[i:i+BATCH_SIZE]

    embeddings = model.encode(batch_texts).tolist()
    to_upsert = list(zip(batch_ids, embeddings, batch_metas))

    index.upsert(vectors=to_upsert)

print("✅ Ingestion complete using SentenceTransformers!")
