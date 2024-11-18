# import pickle
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

import os
import pickle
import numpy as np
import faiss

def create_and_store_vector_store(docs, file_path):
    new_embeddings = [embedding_model.encode(doc.page_content, convert_to_numpy=True) for doc in docs]

    # Check if the file already exists
    if os.path.exists(file_path):
        # Load existing FAISS index and documents
        with open(file_path, 'rb') as f:
            existing_faiss_index, existing_docs = pickle.load(f)
        # Add new embeddings to the existing index
        existing_faiss_index.add(np.array(new_embeddings))
        # Combine the new documents with the existing ones
        updated_docs = existing_docs + docs
    else:
        # Create a new FAISS index if the file does not exist
        existing_faiss_index = faiss.IndexFlatL2(new_embeddings[0].shape[0])
        existing_faiss_index.add(np.array(new_embeddings))
        updated_docs = docs

    # Save updated index and documents to disk
    with open(file_path, 'wb') as f:
        pickle.dump((existing_faiss_index, updated_docs), f)

    return existing_faiss_index


def load_vector_store(file_path):
    with open(file_path, 'rb') as f:
        faiss_index, docs = pickle.load(f)
    return faiss_index, docs