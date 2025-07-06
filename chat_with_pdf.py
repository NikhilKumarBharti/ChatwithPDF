# chat_with_pdf.py
import os
import logging
from typing import List, Optional, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompts import PromptTemplate
from pydantic import Field
import time
import requests
import pickle

logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """Custom OpenRouter LLM wrapper for modern LangChain"""
    
    api_key: str = Field(...)
    model: str = Field(default="deepseek/deepseek-chat-v3-0324:free")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    
    class Config:
        """Pydantic configuration"""
        extra = "forbid"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make API call to OpenRouter"""
        return self._generate_text(prompt)
    
    def _generate_text(self, prompt: str) -> str:
        """Generate response from OpenRouter API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "ChatWithPDF"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Use the API URL directly as a string
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            
            response = requests.post(
                api_url, 
                json=data, 
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected API response format: {result}")
                return "I apologize, but I received an unexpected response format from the API."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return "I apologize, but I'm having trouble connecting to the API right now."
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class ChatWithPDF:
    def __init__(self):
        """Initialize the ChatWithPDF system with all required components"""
        self.vector_store = None
        self.chat_chain = None
        self.memory = None
        self.llm = None
        self.embeddings = None
        self.vector_store_path = "vector_store"
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_memory()
        self._load_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        try:
            embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            
            model_kwargs = {}
            if os.getenv('EMBEDDING_DEVICE'):
                model_kwargs['device'] = os.getenv('EMBEDDING_DEVICE')
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Successfully initialized embeddings with model: {embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize OpenRouter LLM"""
        try:
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            model_name = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
            
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key not found in environment variables")
            
            # Create custom OpenRouter LLM instance with proper parameter passing
            self.llm = OpenRouterLLM(
                api_key=openrouter_api_key,
                model=model_name,
                temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
                max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000'))
            )
            
            logger.info(f"Successfully initialized OpenRouter LLM with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            logger.info("Successfully initialized conversation memory")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise
    
    def _load_vector_store(self):
        """Load existing vector store if it exists"""
        try:
            # Check if the vector store directory exists and contains the required files
            if os.path.exists(self.vector_store_path) and os.path.isdir(self.vector_store_path):
                # Check if the directory contains FAISS files
                index_file = os.path.join(self.vector_store_path, "index.faiss")
                pkl_file = os.path.join(self.vector_store_path, "index.pkl")
                
                if os.path.exists(index_file) and os.path.exists(pkl_file):
                    self.vector_store = FAISS.load_local(
                        self.vector_store_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Loaded existing vector store")
                else:
                    logger.info("Vector store directory exists but is incomplete, will recreate")
                    # Clean up incomplete vector store
                    import shutil
                    shutil.rmtree(self.vector_store_path)
                    self.vector_store = None
            else:
                logger.info("No existing vector store found")
                self.vector_store = None
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            # Clean up corrupted vector store
            if os.path.exists(self.vector_store_path):
                import shutil
                if os.path.isdir(self.vector_store_path):
                    shutil.rmtree(self.vector_store_path)
                elif os.path.isfile(self.vector_store_path):
                    os.remove(self.vector_store_path)
                logger.info("Removed corrupted vector store")
            self.vector_store = None
    
    def _save_vector_store(self):
        """Save the current vector store"""
        try:
            if self.vector_store is not None:
                # Remove existing vector store directory if it exists
                if os.path.exists(self.vector_store_path):
                    import shutil
                    if os.path.isdir(self.vector_store_path):
                        shutil.rmtree(self.vector_store_path)
                    elif os.path.isfile(self.vector_store_path):
                        os.remove(self.vector_store_path)
                
                self.vector_store.save_local(self.vector_store_path)
                logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def process_pdf(self, pdf_path: str, replace_existing: bool = True) -> bool:
        """Process a PDF file and store it in the vector database"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                logger.error("No documents loaded from PDF")
                return False
            
            # Split documents into chunks
            chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
            chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
            texts = text_splitter.split_documents(documents)
            logger.info(f"Split PDF into {len(texts)} chunks")
            
            # Create or update vector store using FAISS
            try:
                if self.vector_store is None or replace_existing:
                    # Create new FAISS vector store (replaces existing)
                    self.vector_store = FAISS.from_documents(
                        documents=texts,
                        embedding=self.embeddings
                    )
                    # Clear chat history when replacing documents
                    self.clear_chat_history()
                    logger.info("Created new vector store (replaced existing)")
                else:
                    # Add new documents to existing vector store
                    new_vector_store = FAISS.from_documents(
                        documents=texts,
                        embedding=self.embeddings
                    )
                    self.vector_store.merge_from(new_vector_store)
                    logger.info("Added documents to existing vector store")
                
                # Save the vector store
                self._save_vector_store()
                
            except Exception as e:
                logger.error(f"Vector store operation failed: {e}")
                return False
            
            # Initialize or update the chat chain
            self._initialize_chat_chain()
            
            logger.info("PDF processed and stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return False
    
    def _initialize_chat_chain(self):
        """Initialize the conversational retrieval chain"""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            
            # Configure retriever
            search_k = int(os.getenv('RETRIEVER_K', '3'))
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_k}
            )
            
            # Create the conversational retrieval chain
            self.chat_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("Chat chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat chain: {e}")
            raise
    
    def chat(self, question: str) -> str:
        """Chat with the PDF content"""
        try:
            if self.chat_chain is None:
                return "Please upload a PDF first before asking questions."
            
            logger.info(f"Processing question: {question}")
            
            # Get response from the chain
            response = self.chat_chain.invoke({"question": question})
            
            answer = response.get("answer", "I couldn't find an answer to your question.")
            
            logger.info("Question processed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "Sorry, I encountered an error while processing your question."
    
    def get_chat_history(self) -> List[str]:
        """Get the chat history"""
        try:
            if self.memory and hasattr(self.memory, 'chat_memory'):
                return [str(msg) for msg in self.memory.chat_memory.messages]
            return []
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def clear_chat_history(self):
        """Clear the chat history"""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("Chat history cleared")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
    
    def clear_vector_store(self):
        """Clear the vector store and reset the system"""
        try:
            # Clear the vector store
            self.vector_store = None
            self.chat_chain = None
            
            # Remove the saved vector store files
            if os.path.exists(self.vector_store_path):
                import shutil
                if os.path.isdir(self.vector_store_path):
                    shutil.rmtree(self.vector_store_path)
                elif os.path.isfile(self.vector_store_path):
                    os.remove(self.vector_store_path)
            
            # Clear chat history
            self.clear_chat_history()
            
            logger.info("Vector store and chat history cleared")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def get_vector_store_info(self) -> dict:
        """Get information about the vector store"""
        try:
            if self.vector_store is None:
                return {"status": "No documents uploaded"}
            
            embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            llm_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
            
            return {
                "status": "Ready",
                "vector_store_type": "FAISS",
                "embedding_model": f"HuggingFace: {embedding_model}",
                "llm_model": f"OpenRouter: {llm_model}"
            }
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {"status": "Error", "error": str(e)}