"""
server.py - FastAPI Backend Server
===================================
This is the main server that handles chat requests using RAG.

Request Flow:
1. User sends message from frontend (POST /chat)
2. FastAPI receives and validates request
3. LangChain retrieves relevant context from FAISS
4. Google Gemini generates answer based on context
5. Response sent back to frontend as JSON

The vector database is loaded ONCE on startup for efficiency.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  # Changed from PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))

# Validate API key
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "AIzaSyAfNdMBZCm4Yqi22hZ9myv2M4hdvWMwvs8":
    raise ValueError(
        "❌ GOOGLE_API_KEY not found in .env file!\n"
        "Get your free API key from: https://makersuite.google.com/app/apikey\n"
        "Then add it to .env file: GOOGLE_API_KEY=your_key_here"
    )

# Initialize FastAPI app
app = FastAPI(
    title="FAST NUCES RAG Chatbot API",
    description="University FAQ Chatbot using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (loaded once on startup)
qa_chain = None
embeddings = None


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the admission requirements?"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    sources: Optional[list] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The admission requirements include...",
                "sources": ["Document chunk 1", "Document chunk 2"]
            }
        }


def load_vector_store():
    """
    Load the FAISS vector store from disk.
    This is called once during server startup.
    
    Returns:
        FAISS: Loaded vector store instance
    """
    print("🗄️  Loading FAISS vector database...")
    
    # Check if index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"❌ FAISS index not found at {FAISS_INDEX_PATH}!\n"
            f"Please run 'python ingest.py' first to create the knowledge base."
        )
    
    # Initialize embeddings (same as used during ingestion)
    global embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load vector store from disk
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS
    )
    
    print(f"✅ Vector database loaded successfully")
    return vectorstore


def create_qa_chain(vectorstore):
    """
    Create the Question-Answering chain with Gemini.
    
    This chain:
    1. Retrieves relevant context from vector store
    2. Sends context + question to Gemini
    3. Returns generated answer
    
    Args:
        vectorstore (FAISS): Vector store instance
    
    Returns:
        RetrievalQA: Configured QA chain
    """
    print("🤖 Initializing Gemini LLM...")
    
    # Initialize Google Gemini
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,  # Lower = more focused, Higher = more creative
        convert_system_message_to_human=True
    )
    
    # Create custom prompt template
    # This guides how Gemini should answer questions
    prompt_template = """You are a helpful assistant for FAST NUCES Islamabad University. 
Use the following context to answer the student's question accurately and concisely.
If the information is not in the context, politely say you don't have that specific information and suggest contacting the administration.

Context:
{context}

Question: {question}

Answer (be specific and helpful):"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    # This automatically:
    # 1. Converts question to embedding
    # 2. Searches for top K similar chunks
    # 3. Passes chunks to LLM
    # 4. Returns generated answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means pass all retrieved docs to LLM
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}  # Retrieve top 3 most relevant chunks
        ),
        return_source_documents=True,  # Include source chunks in response
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print(f"✅ QA chain initialized (retrieving top {RETRIEVAL_K} documents)")
    return qa_chain


@app.on_event("startup")
async def startup_event():
    """
    Run once when server starts.
    Loads the vector database and initializes the QA chain.
    """
    print("\n" + "=" * 60)
    print("🚀 Starting FAST NUCES RAG Chatbot Server")
    print("=" * 60)
    
    global qa_chain
    
    try:
        # Load vector store
        vectorstore = load_vector_store()
        
        # Create QA chain
        qa_chain = create_qa_chain(vectorstore)
        
        print("\n" + "=" * 60)
        print("✅ Server ready to handle requests!")
        print("=" * 60)
        print(f"📡 API: http://localhost:8000/chat")
        print(f"🌐 Frontend: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Startup Error: {str(e)}")
        raise


@app.get("/")
async def serve_frontend():
    """
    Serve the main HTML page.
    """
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Receive user message
    2. Query vector database for relevant context
    3. Send context + question to Gemini
    4. Return generated answer
    
    Args:
        request (ChatRequest): User's message
    
    Returns:
        ChatResponse: Generated answer with sources
    """
    try:
        # Validate message
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        print(f"\n💬 Received question: {request.message}")
        
        # Query the RAG chain
        # This automatically:
        # 1. Retrieves relevant documents
        # 2. Generates answer using Gemini
        result = qa_chain.invoke({"query": request.message})
        
        # Extract answer
        answer = result["result"]
        
        # Extract source documents (for debugging/transparency)
        source_docs = result.get("source_documents", [])
        sources = [doc.page_content[:200] + "..." for doc in source_docs]
        
        print(f"✅ Generated answer (used {len(source_docs)} sources)")
        
        return ChatResponse(
            response=answer,
            sources=sources if sources else None
        )
    
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify server is running.
    """
    return {
        "status": "healthy",
        "message": "FAST NUCES RAG Chatbot is running",
        "vector_db_loaded": qa_chain is not None
    }


# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True  # Auto-reload on code changes (disable in production)
    )