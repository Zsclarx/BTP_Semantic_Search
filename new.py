import os
import PyPDF2
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from joblib import Parallel, delayed

def parse_pdf(file_path):
    """
    Extract paragraphs from a PDF with page references.
    
    Args:
        file_path (str): Path to the PDF file.
    Returns:
        list: List of tuples (paragraph_text, page_number).
    """
    paragraphs_with_page_refs = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                # Split by double newlines and filter out empty paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for para in paragraphs:
                    paragraphs_with_page_refs.append((para, page_num))
    return paragraphs_with_page_refs

def chunk_text_by_paragraphs(paragraphs_with_refs, max_chunk_size=256):
    """
    Chunk paragraphs, splitting those exceeding max_chunk_size.
    
    Args:
        paragraphs_with_refs (list): List of (paragraph, page_num) tuples.
        max_chunk_size (int): Maximum number of tokens per chunk.
    Returns:
        list: List of (chunk_text, [page_nums]) tuples.
    """
    chunks = []
    for para, page_num in paragraphs_with_refs:
        words = para.split()
        if len(words) <= max_chunk_size:
            chunks.append((para, [page_num]))
        else:
            # Split long paragraphs into smaller chunks
            start = 0
            while start < len(words):
                end = start + max_chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append((chunk, [page_num]))
                start = end
    return chunks

@st.cache_data(show_spinner=False)
def get_embeddings_parallel(text_chunks, _model, batch_size=32, n_jobs=-1):
    """
    Generate embeddings for text chunks in parallel.
    
    Args:
        text_chunks (list): List of (chunk_text, page_nums) tuples.
        _model: SentenceTransformer model instance.
        batch_size (int): Number of chunks per batch.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).
    Returns:
        np.ndarray: Array of embeddings.
    """
    texts = [chunk[0] for chunk in text_chunks]
    def process_batch(batch):
        return _model.encode(batch, convert_to_numpy=True)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    embeddings_list = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in batches
    )
    return np.vstack(embeddings_list)

@st.cache_data(show_spinner=False)
def create_faiss_index(embeddings):
    """
    Create a FAISS index for fast similarity searches.
    
    Args:
        embeddings (np.ndarray): Array of chunk embeddings.
    Returns:
        faiss.Index: Trained FAISS index.
    """
    dimension = embeddings.shape[1]
    nlist = min(100, len(embeddings) // 4)  # Number of clusters
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(10, nlist)  # Number of clusters to search
    return index

def semantic_search(query, index, text_chunks, _model):
    """
    Perform semantic search to find top matching chunks.
    
    Args:
        query (str): User query.
        index (faiss.Index): FAISS index of embeddings.
        text_chunks (list): List of (chunk_text, page_nums) tuples.
        _model: SentenceTransformer model instance.
    Returns:
        list: List of (chunk_text, page_nums, distance) tuples.
    """
    query_embedding = _model.encode([query], convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, 10)  # Top 10 results
    results = []
    for j, i in enumerate(indices[0]):
        if i < len(text_chunks):
            chunk_text, pages = text_chunks[i]
            results.append((chunk_text, pages, distances[0][j]))
    return results

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the SentenceTransformer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit Interface
st.title("Generalized Semantic Search for PDF Documents")

uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Extracting text from PDF...")
    paragraphs_with_refs = parse_pdf("temp.pdf")

    st.write("Chunking document...")
    model = load_model()
    text_chunks = chunk_text_by_paragraphs(paragraphs_with_refs, max_chunk_size=256)

    st.write("Generating embeddings...")
    embeddings = get_embeddings_parallel(text_chunks, model, batch_size=32, n_jobs=-1)

    st.write("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    query = st.text_input("Enter your query:")

    if query:
        st.write("Searching...")
        results = semantic_search(query, index, text_chunks, model)

        st.write("### Top Results:")
        for i, (chunk_text, pages, score) in enumerate(results):
            with st.expander(f"Result {i+1} (Similarity Score: {score:.4f})"):
                st.write(f"**Text:** {chunk_text}")
                st.write(f"**Pages:** {', '.join(map(str, pages))}")
        st.success("Search completed!")

    # Clean up temporary file
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")