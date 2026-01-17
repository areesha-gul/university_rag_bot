"""
ingest.py - Knowledge Base Ingestion Script (PDF Version)
==========================================================
This script builds the vector database from a PDF document.

Flow:
1. Load PDF document
2. Split into chunks for better retrieval
3. Generate embeddings using local HuggingFace model
4. Store in FAISS vector database
5. Save to disk for server usage
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports - UPDATED FOR PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Changed from TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configuration - UPDATED PATH
DATA_PATH = "data/university_handbook.pdf"  # Changed from .txt to .pdf
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_documents():
    """
    Load documents from the PDF file.
    
    Returns:
        list: List of Document objects
    """
    print(f"📄 Loading PDF document from {DATA_PATH}...")
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"PDF file not found at {DATA_PATH}. "
            f"Please place your PDF file there."
        )
    
    # Load PDF file - UPDATED LOADER
    # PyPDFLoader automatically extracts text from each page
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} page(s) from PDF")
    
    # Show preview of first page
    if documents:
        preview = documents[0].page_content[:200].replace('\n', ' ')
        print(f"   Preview: {preview}...")
    
    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    RecursiveCharacterTextSplitter is better for PDFs as it:
    - Tries to keep paragraphs together
    - Handles various text structures
    - Respects natural text boundaries
    
    Args:
        documents (list): List of Document objects
    
    Returns:
        list: List of chunked Document objects
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    
    # Initialize text splitter - UPDATED SPLITTER
    # RecursiveCharacterTextSplitter is optimal for PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",  # First try double newlines (paragraphs)
            "\n",    # Then single newlines
            ". ",    # Then sentences
            " ",     # Then words
            ""       # Finally characters
        ],
        is_separator_regex=False
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show statistics
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    print(f"   Average chunk size: {int(avg_chunk_size)} characters")
    
    return chunks


def create_embeddings():
    """
    Initialize the embedding model.
    This uses a local HuggingFace model (no API calls needed).
    
    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    print(f"\n🧠 Initializing embedding model: {EMBEDDING_MODEL}")
    print("   This will download the model on first run (~100MB)")
    print("   Subsequent runs will use cached model")
    
    # Initialize embeddings
    # This model runs locally and converts text to 384-dimensional vectors
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU (works on all machines)
        encode_kwargs={'normalize_embeddings': True}  # Better similarity search
    )
    
    print("✅ Embedding model loaded")
    return embeddings


def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store from document chunks.
    
    Args:
        chunks (list): List of document chunks
        embeddings: Embedding model instance
    
    Returns:
        FAISS: Vector store instance
    """
    print(f"\n🗄️  Creating FAISS vector database...")
    print(f"   Processing {len(chunks)} chunks...")
    print("   This may take 1-2 minutes for large PDFs...")
    
    # Create FAISS index from documents
    # FAISS (Facebook AI Similarity Search) is optimized for fast similarity search
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    print("✅ Vector database created")
    return vectorstore


def save_vector_store(vectorstore):
    """
    Save vector store to disk for later use by server.
    
    Args:
        vectorstore (FAISS): Vector store instance
    """
    print(f"\n💾 Saving vector database to {FAISS_INDEX_PATH}...")
    
    # Create directory if it doesn't exist
    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    
    # Save to disk
    # This saves both the index and the original documents
    vectorstore.save_local(FAISS_INDEX_PATH)
    
    print("✅ Vector database saved successfully")


def main():
    """
    Main ingestion pipeline for PDF processing.
    """
    print("=" * 60)
    print("🚀 FAST NUCES RAG Chatbot - PDF Knowledge Base Ingestion")
    print("=" * 60)
    
    try:
        # Step 1: Load PDF document
        documents = load_documents()
        
        # Step 2: Split into chunks (using RecursiveCharacterTextSplitter)
        chunks = split_documents(documents)
        
        # Step 3: Initialize embeddings
        embeddings = create_embeddings()
        
        # Step 4: Create vector store
        vectorstore = create_vector_store(chunks, embeddings)
        
        # Step 5: Save to disk
        save_vector_store(vectorstore)
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ SUCCESS! PDF knowledge base is ready.")
        print("=" * 60)
        print(f"📊 Statistics:")
        print(f"   - PDF pages: {len(documents)}")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Embedding dimension: 384")
        print(f"   - Storage location: {FAISS_INDEX_PATH}/")
        print(f"\n🎯 Next step: Run 'python server.py' to start the chatbot")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Solution:")
        print(f"   1. Place your PDF file at: {DATA_PATH}")
        print(f"   2. Make sure the file is named 'university_handbook.pdf'")
        print(f"   3. Run this script again")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        print("\n💡 Common issues:")
        print("   - PDF might be corrupted (try opening in Adobe Reader)")
        print("   - PDF might be scanned image (needs OCR)")
        print("   - PDF might be password protected")
        raise


if __name__ == "__main__":
    main()