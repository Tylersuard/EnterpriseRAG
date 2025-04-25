import PyPDF2
import os


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def load_pdfs_from_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(file_path)
            texts.append(text)
    return texts


def fixed_size_chunking(text, chunk_size=128):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def fixed_size_chunking_with_overlap(text, chunk_size=128, overlap=30):

    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def break_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def calculate_cosine_similarity(sentences, model, batch_size=8):

    sentence_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_embeddings = model.encode(sentences[i : i + batch_size])
        sentence_embeddings.extend(batch_embeddings)
    sentence_embeddings = np.array(sentence_embeddings)
    cosine_similarities = cosine_similarity(sentence_embeddings)

    return cosine_similarities


from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import sent_tokenize, TextTilingTokenizer


def find_threshold(cosine_similarities, percentile=25):
    similarities = cosine_similarities.flatten()
    threshold = np.percentile(similarities, percentile)
    return threshold


def anchor_based_segmentation(sentences, cosine_similarities, adaptive_percentile=25):
    threshold = find_threshold(cosine_similarities, adaptive_percentile)
    chunks = []
    current_chunk = [sentences[0]]
    anchor_idx = 0
    for i in range(1, len(sentences)):
        similarity = cosine_similarities[anchor_idx, i]
        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            anchor_idx = i
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# embeddings = []
# for chunk in chunks:
#     embedding = model.encode(chunk)
#     embeddings.append(embedding)


from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from openai import OpenAI

qdrant_client = QdrantClient(
    url="QDRANT_DATABASE_URL",
    api_key="QDRANT_DATABASE_API_KEY",
)

openai_client = OpenAI(api_key="OPENAI_API_KEY")


def generate_keywords(chunk):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use the desired GPT model
            messages=[
                {
                    "role": "system",
                    "content": """You are a tool that extracts concise
keywords from a given text. Respond only with a plain list of
keywords, each separated by commas.""",
                },
                {
                    "role": "user",
                    "content": f"Extract keywords from this text chunk:\n{chunk}",
                },
            ],
            temperature=0.5,
        )
        keywords = response.choices[0].message.content.strip()
        keywords_list = [
            keyword.strip() for keyword in keywords.split(",") if keyword.strip()
        ]
        return keywords_list
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return []


from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

qdrant_client = QdrantClient(
    url="QDRANT_URL",
    api_key="QDRANT_API_KEY",
)


def ingest_chunks_with_metadata_to_qdrant(all_chunks):
    try:
        collections = qdrant_client.get_collections()
        if "test_collection" not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name="test_collection",
                vectors_config={"size": 768, "distance": "Cosine"},
            )
    except Exception as e:
        print("Error in collection handling:", e)
        return

    points = []
    for i, item in enumerate(all_chunks):
        chunk = item["chunk"]
        keywords_list = generate_keywords(chunk)
        embedding = model.encode(chunk)

        point = PointStruct(
            id=i,
            vector=embedding,
            payload={"chunk": chunk, "keywords_list": keywords_list},
        )
        points.append(point)


from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

qdrant_client = QdrantClient(
    url="QDRANT_DATABASE_URL",
    api_key="QDRANT_DATABASE_API_KEY",
)


def ingest_chunks_without_metadata_to_qdrant(all_chunks):
    try:
        collections = qdrant_client.get_collections()
        if "test_collection" not in [col.name for col in collections.collections]:
            qdrant_client.create_collection(
                collection_name="test_collection",
                vectors_config={"size": 768, "distance": "Cosine"},
            )
    except Exception as e:
        print("Error in collection handling:", e)
        return

    points = []
    for i, item in enumerate(all_chunks):
        chunk = item["chunk"]
        embedding = model.encode(chunk)

        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk": chunk,
            },
        )
        points.append(point)

    try:
        qdrant_client.upsert(collection_name="test_collection", points=points)
        print(
            f"Ingested {len(points)} chunks into Qdrant collection 'test_collection' without metadata."
        )
    except Exception as e:
        print(f"Error ingesting chunks: {e}")


if __name__ == "__main__":
    pdf_path = "my_document.pdf"
    CHUNK_SIZE = 500
    OVERLAP = 50

    pdf_text = extract_text_from_pdf(pdf_path)

    chunks = fixed_size_chunking_with_overlap(
        pdf_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    )

    all_chunks = [{"chunk": c} for c in chunks]

    ingest_chunks_without_metadata_to_qdrant(all_chunks)

    print(f"Done. Ingested {len(all_chunks)} chunks from {pdf_path} into Qdrant.")
