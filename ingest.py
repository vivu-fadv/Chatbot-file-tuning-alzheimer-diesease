from pathlib import Path
import os

# Step 0: Load environment variables from .env when available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
HF_TOKEN = os.getenv('HF_TOKEN')

# Create vector database 
def create_vector_db():
    # Step 1: Find all PDFs to ingest.
    # Collect every PDF in data/ so all documents are indexed.
    data_dir = Path(DATA_PATH)
    pdf_paths = sorted(data_dir.glob('*.pdf'))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{data_dir}'.")

    documents = []
    for pdf_file_path in pdf_paths:
        # Step 2: Read each PDF and append its pages to one document list.
        # Load each PDF into LangChain Document objects.
        loader = PyPDFLoader(str(pdf_file_path))
        documents.extend(loader.load())

    # Step 3: Split long text into smaller chunks for semantic search.
    # Split long pages into overlapping chunks for better retrieval quality.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embedding_kwargs = {'device': 'cpu'}
    if HF_TOKEN:
        embedding_kwargs['token'] = HF_TOKEN

    # Step 4: Convert chunks into vectors (embeddings).
    # Create embeddings and store vectors in a local FAISS index.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs=embedding_kwargs)

    # Step 5: Save vectors to disk so chatbot.py can retrieve from this index.
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()