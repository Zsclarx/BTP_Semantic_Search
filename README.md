-----

# Semantic Search Engine for Web Scraped Data

## 📖 About The Project

This project is a sophisticated Semantic Search Engine designed to understand the context and meaning behind a user's query, rather than relying on simple keyword matching. It is built using a powerful Sentence Transformer model (`all-MiniLM-L6-v2`) to create vector embeddings of text data, enabling it to find the most contextually relevant information from a scraped dataset.

The application is built with Streamlit, providing a clean and intuitive web interface for users to input queries and view results. The backend handles data loading, embedding generation, and similarity search to deliver accurate and meaningful search outcomes.

-----

## ✨ Features

  * **Semantic Understanding:** Goes beyond keywords to grasp the intent behind your search query.
  * **Intuitive UI:** A simple and user-friendly web interface built with Streamlit.
  * **High Accuracy:** Utilizes state-of-the-art sentence embeddings for precise results.
  * **Efficient Search:** Employs pre-computed embeddings and FAISS for fast similarity lookups.
  * **Scalable:** The architecture allows for easy expansion with larger datasets.

-----

## 🛠️ Technology Stack

This project is built with a modern stack for machine learning and web development:

  * **Backend:** Python
  * **ML/NLP:** PyTorch, Sentence Transformers, Transformers
  * **Vector Search:** FAISS (Facebook AI Similarity Search)
  * **Web Framework:** Streamlit
  * **Data Handling:** Pandas, NumPy

-----

## ⚙️ How It Works

The engine's workflow can be broken down into two main phases:

1.  **Indexing Phase (Offline):**

      * **Data Loading:** The text data is loaded from the `scraped data` directory.
      * **Embedding Generation:** The `all-MiniLM-L6-v2` model from Sentence Transformers is used to convert each piece of text data into a high-dimensional vector (embedding).
      * **Indexing:** These embeddings are stored in a FAISS index, a highly efficient library for similarity search. This index is saved to disk for quick access during search.

2.  **Search Phase (Online):**

      * **User Query:** The user enters a natural language query into the Streamlit web interface.
      * **Query Embedding:** The same Sentence Transformer model converts the user's query into a vector embedding.
      * **Similarity Search:** FAISS is used to compare the query embedding against all the indexed document embeddings, retrieving the top 'k' most similar vectors.
      * **Display Results:** The application retrieves the original text corresponding to the top results and displays them to the user.

-----

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/Zsclarx/BTP_Semantic_Search.git
    cd BTP_Semantic_Search
    ```

2.  **Create a virtual environment (recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Usage

Once the installation is complete, you can run the Streamlit application with a single command:

```sh
streamlit run app.py
```

Navigate to the local URL provided by Streamlit in your browser to start using the search engine.
