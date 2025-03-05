import os
import PyPDF2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from sentence_transformers import SentenceTransformer
import faiss
from joblib import Parallel, delayed

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flash messages
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Global variables to store index, text chunks, and PDF path
index = None
text_chunks = None
pdf_path = None

def parse_pdf(file_path):
    """Extract text from PDF and associate paragraphs with page numbers."""
    paragraphs_with_page_refs = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for para in paragraphs:
                    paragraphs_with_page_refs.append((para, page_num))
    return paragraphs_with_page_refs

def chunk_text_by_paragraphs(paragraphs_with_refs, max_chunk_size=256):
    """Chunk paragraphs into smaller pieces if needed, with page references."""
    chunks = []
    for para, page_num in paragraphs_with_refs:
        words = para.split()
        if len(words) <= max_chunk_size:
            chunks.append((para, [page_num]))
        else:
            start = 0
            while start < len(words):
                end = start + max_chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append((chunk, [page_num]))
                start = end
    return chunks

def get_embeddings_parallel(text_chunks, model, batch_size=32, n_jobs=-1):
    """Generate embeddings for text chunks in parallel."""
    texts = [chunk[0] for chunk in text_chunks]
    def process_batch(batch):
        return model.encode(batch, convert_to_numpy=True)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    embeddings_list = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in batches
    )
    return np.vstack(embeddings_list)

def create_faiss_index(embeddings):
    """Create a FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]
    nlist = min(100, len(embeddings) // 4)
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(10, nlist)
    return index

def semantic_search(query, index, text_chunks, model):
    """Perform semantic search and return top 10 results with page numbers."""
    query_embedding = model.encode([query], convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, 10)
    results = []
    for j, i in enumerate(indices[0]):
        if i < len(text_chunks):
            chunk_text, pages = text_chunks[i]
            results.append((chunk_text, pages))
    return results

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global index, text_chunks, pdf_path
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.pdf')
            file.save(filepath)
            # Process the PDF to create the index
            paragraphs_with_refs = parse_pdf(filepath)
            text_chunks = chunk_text_by_paragraphs(paragraphs_with_refs)
            embeddings = get_embeddings_parallel(text_chunks, model)
            index = create_faiss_index(embeddings)
            pdf_path = '/static/uploads/uploaded.pdf'
            flash('PDF processed successfully. You can now enter queries.')
            return redirect(url_for('query'))
        else:
            flash('Invalid file type')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    global index, text_chunks, pdf_path
    if index is None:
        flash('Please upload a PDF first.')
        return redirect(url_for('upload'))
    if request.method == 'POST':
        query_text = request.form['query']
        results = semantic_search(query_text, index, text_chunks, model)
        return render_template('query.html', results=results, pdf_path=pdf_path)
    return render_template('query.html', results=None)




if __name__ == '__main__':
    app.run(debug=True)