import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

client_model= OpenAI()
def get_embedding(text, model=EMBEDDING_MODEL):
    response = client_model.embeddings.create(
        input=text,
        model=model,
        encoding_format="float"
    )
    return response.data[0].embedding


def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sum(np.abs(vec1 - vec2))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def normalize_column(values):
    scaler = MinMaxScaler()
    return scaler.fit_transform([values]).flatten()