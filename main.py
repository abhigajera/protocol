from __future__ import annotations
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing import List, Annotated, Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
# Using HTTPBearer for simple token auth, removing complex OAuth2 fields
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
from pydantic import BaseModel
from jose import JWTError, jwt
# REMOVED: CryptContext import (since we are not using hashing)

# LangChain and RAG Imports (Original Gemini setup)
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sentence_transformers 


# --- 1. Configuration and Component Initialization ---
load_dotenv()

# --- Credentials ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- JWT Settings ---
JWT_SECRET = os.getenv("JWT_SECRET") 
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, JWT_SECRET]):
    raise ValueError("Missing one or more required environment variables.")

# --- Supabase & Embeddings ---
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# --- LLM and RAG Chain (Using Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the user's question STRICTLY and concisely 
    based only on the context provided below. If the answer is not in the context, 
    state that you cannot find the answer.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
""")
RAG_CHAIN = RAG_PROMPT | llm | StrOutputParser()


# =================================================================
# 2. SIMPLE JWT AUTHENTICATION SETUP AND PYDANTIC MODELS
# =================================================================

# Security Dependency for 'Bearer' token validation
jwt_bearer = HTTPBearer() 
# REMOVED: pwd_context = CryptContext(...)

# Pydantic Models 
class LoginRequest(BaseModel): # Custom model for simple login JSON body
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class QueryRequest(BaseModel):
    query: str
    k: int = 4 

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Dummy Database for User Auth - NOW STORES PLAIN TEXT PASSWORD
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        # CHANGED: Storing the plain text password directly
        "password": "admin123" 
    }
}

# REMOVED: def verify_password(...)

def authenticate_user(username: str, password: str):
    """Simulates checking credentials against a database using direct comparison."""
    user = FAKE_USERS_DB.get(username)
    if not user:
        return False
    # CHANGED: Direct comparison of the plaintext password
    if user["password"] != password: 
        return False
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    """Creates a signed JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "sub": str(data["sub"])})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM) 
    return encoded_jwt

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(jwt_bearer)):
    """Dependency to decode and validate the JWT from the Authorization header."""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM]) 
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Check if user exists 
        if username not in FAKE_USERS_DB:
            raise credentials_exception
        
        return username
    except JWTError:
        raise credentials_exception

# =================================================================
# 3. FastAPI Endpoints
# =================================================================

app = FastAPI(title="Gemini RAG Chatbot API with Simple JWT Auth (DEV MODE)")

# --- Login Endpoint (Simple JSON Body) ---
@app.post("/login", response_model=Token)
async def login_for_access_token(request: LoginRequest):
    """Generates a JWT token using only a username and password in a JSON body."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- Protected Endpoints ---

@app.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    current_user: Annotated[str, Depends(verify_jwt_token)] = None # JWT Protection
):
    """Handles PDF file upload, chunking, embedding, and storage in Supabase."""
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        # 1. Save the file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load the document and split it into chunks
        loader = PyPDFLoader(str(temp_file_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks: List[Document] = splitter.split_documents(documents)
        
        # Add metadata source for citation
        for chunk in chunks:
            chunk.metadata['source'] = file.filename
        
        # 3. Embed and store chunks in Supabase
        vector_store.add_documents(chunks)

        return {
            "message": f"File processed and indexed successfully by user: {current_user}.",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    finally:
        # 4. Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    current_user: Annotated[str, Depends(verify_jwt_token)] = None
):
    """Retrieves relevant context from Supabase using custom RPC + Gemini."""
    
    try:
        # --- 1. Create embedding for the query ---
        embedding = embeddings.embed_query(request.query)

        # --- 2. Call Supabase RPC manually ---
        response = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_threshold": 0.0,
                "match_count": request.k,
                "filter": {}
            }
        ).execute()

        print("RPC Response:", response)

        results = response.data
        if not results:
            return {"answer": "No relevant documents found.", "sources": []}

        # --- 3. Extract context + sources ---
        context = "\n\n".join([r["content"] for r in results])
        sources = list({r["metadata"].get("source", "Unknown") for r in results})

        # --- 4. Generate RAG answer from Gemini ---
        answer = RAG_CHAIN.invoke({"context": context, "question": request.query})

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        print("Error during query:", e)
        raise HTTPException(status_code=500, detail="Failed to process query")
