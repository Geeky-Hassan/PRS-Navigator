import os
import time
import streamlit as st
from dotenv import load_dotenv

import warnings

# Langchain and Vector Store Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader

# Conversational Memory Imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Additional imports
import shutil

# Configuration and Environment Setup
load_dotenv()

class PRSChatbot:
    def __init__(self):
        """
        Initialize the PRS RAG Chatbot with direct configuration
        """
        # Hardcoded configuration
        self.config_data = {
            "INSTITUTION_NAME": "University of Management and Technology",
            "PRS_LOCATION": "Admin Building, Level-1, C-II, Joher Town, Lahore, Pakistan",
            "PRS_PHONE": "042-111-300-200",
            "PRS_EXTENSIONS": "3749, 3713",
            "PRS_EMAIL": "prshelpdesk@umt.edu.pk"
        }
        
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.vectors_dir = os.path.join(self.working_dir, "vectors_gemini")
        
        os.makedirs(self.vectors_dir, exist_ok=True)
        
        self._setup_environment()
        
        self._initialize_session_state()
        
        # Set up LLM and Embedding models
        self._setup_models()
    
    def _setup_environment(self):
        """
        Set up environment variables
        """
        # Set Google API Key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
    
    def _initialize_session_state(self):
        """
        Initialize Streamlit session state variables
        """
        # Initialize chat history if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize vectors if not exists
        if "vectors" not in st.session_state:
            st.session_state.vectors = None
        
        # Initialize conversation memory
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )


    def _setup_models(self):
        """
        Set up Language Model and Embedding models with optimized parameters
        """
        # Gemini LLM with enhanced parameters
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,  # Controlled randomness
            top_p=0.9,  # Improved diversity
            top_k=40,   # Increased token selection range
            convert_system_message_to_human=True
        )
        
        # Google Embedding Model with advanced configuration
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            task_type="retrieval_query"
        )
    
    def create_vector_embedding(self, data_directory="./data"):
        """
        Create advanced vector embeddings from PDF documents with local storage
        
        :param data_directory: Directory containing PDF files
        """
        try:
            # Enhanced document loading
            docs = []
            pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
            
            if not pdf_files:
                st.error("No PDF files found in the data directory!")
                return False
            
            for pdf_file in pdf_files:
                file_path = os.path.join(data_directory, pdf_file)
                loader = PDFPlumberLoader(file_path)
                file_docs = loader.load()
                
                # Enhance each document with filename metadata
                for doc in file_docs:
                    doc.metadata['source'] = pdf_file
                
                docs.extend(file_docs)
            
            # Advanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Optimal chunk size
                chunk_overlap=200,  # Improved context preservation
                length_function=len
            )
            final_documents = text_splitter.split_documents(docs)
            
            # Create vector store with enhanced metadata
            vector_store = FAISS.from_documents(
                final_documents, 
                self.embeddings
            )
            
            # Save vector store locally
            vector_store.save_local(self.vectors_dir)
            
            # Update session state
            st.session_state.vectors = vector_store
            
            st.success(f"Vector Store created with {len(final_documents)} document chunks")
            return True
        
        except Exception as e:
            st.error(f"Error creating vector embeddings: {e}")
            return False
    
    def load_vector_embedding(self):
        """
        Load existing vector embeddings from local storage
        
        :return: bool indicating successful loading
        """
        try:
            # Check if local vector store exists
            index_files = [f for f in os.listdir(self.vectors_dir) if f.startswith('index')]
            
            if not index_files:
                st.warning("No existing vector embeddings found. Please generate them first.")
                return False
            
            # Load the local vector store
            vector_store = FAISS.load_local(
                self.vectors_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Update session state
            st.session_state.vectors = vector_store
            
            st.success("Vector embeddings loaded successfully!")
            return True
        
        except Exception as e:
            st.error(f"Error loading vector embeddings: {e}")
            return False
    
    def create_rag_prompt(self):
        """
        Create a comprehensive prompt template for detailed responses.
        
        :return: Configured ChatPromptTemplate
        """
        config = self.config_data
        return ChatPromptTemplate.from_template(f"""
        Role: Detailed PRS Information Assistant for UMT

        Document Context Information:
        - Document Sources:
            1. PRS FAQs from Official Website
            2. Undergraduate Studies Handbook
            3. Graduate Studies (MS/PhD) Handbook

        Important Notes:
        - Policies may have similar names but different internal instructions.
        - ALWAYS specify and refer to the EXACT document source.
        - Distinguish between Undergraduate (BS) and Graduate (MS/PhD) policies.

        Strict Guidelines:
        - You must not give any answer outside of the vector database context.
        - If the question is outside the scope, respond:
        "I am PRS chatbot helper, I can't answer outside of my scope. Please contact PRS for further details."
        - Your only motive is to assist with PRS-related data.

        Language Handling Instructions:
        - If user communicates in Roman Urdu, respond in Roman Urdu.
        - Maintain professional and clear communication.
        - Ensure accurate translation of technical terms.
        - If unable to understand Roman Urdu, ask for clarification in both English and Roman Urdu.

        Additional Language Processing:
        - Detect input language (Roman Urdu or English).
        - Respond in the same language as the input.
        - Use Google Gemini's multilingual capabilities for accurate translation.

        Objective: Provide precise, source-referenced answers from PRS documents.

        Context Guidelines:
        - Use ONLY information from the provided documents.
        - Include specific page and document references.
        - Be concise yet comprehensive.
        - Clearly indicate which document type (UG/Grad) the information is from.

        Context Documentation:
        {{context}}

        Previous Conversation:
        {{chat_history}}

        Current Question: {{input}}

        Response Instructions:
        - Directly answer the question.
        - Cite specific document sources with exact document name.
        - Include page numbers when possible.
        - If information varies between UG/Grad, explain the differences.
        - If information is unavailable, clearly state so.
        - Suggest contacting PRS for further details.

        PRS Contact for Unresolved Queries:
        - Location: {config['PRS_LOCATION']}
        - Phone: {config['PRS_PHONE']}, Ext: {config['PRS_EXTENSIONS']}
        - Email: {config['PRS_EMAIL']}
        """)

    def query_documents(self, user_prompt):
        """
        Advanced document querying with comprehensive reference tracking
        
        :param user_prompt: User's query string
        :return: Detailed response dictionary
        """
        if st.session_state.vectors is None:
            st.warning("Please create or load document embeddings first!")
            return None
        
        try:
            # Create comprehensive prompt with enhanced context handling
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful PRS Information Assistant for UMT. 
            Remember the entire conversation context and user details.
                                                      

            Conversation History:
            {chat_history}

            Context Documentation:
            {context}

            Current Question: {input}

            Provide a comprehensive, context-aware response that:
            - Directly answers the question
            - References previous conversation if relevant
            - Maintains conversation continuity
            - Uses information from provided documents
            """)
            
            # Enhanced retriever configuration
            retriever = st.session_state.vectors.as_retriever(
                search_kwargs={
                    "k": 5,          # Top 5 most relevant documents
                    "fetch_k": 20,   # Initially fetch more documents
                    "search_type": "mmr",  # Maximal Marginal Relevance for diversity
                    "lambda_mult": 0.5  # Balance between relevance and diversity
                }
            )
            
            # Prepare detailed chat history
            # Convert chat history to a format that preserves context
            formatted_chat_history = []
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    formatted_chat_history.append(f"Human: {msg['content']}")
                else:
                    formatted_chat_history.append(f"Assistant: {msg['content']}")
            
            # Join chat history into a single string
            chat_history_str = "\n".join(formatted_chat_history)
            
            # Create document chain
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )
            
            # Process query with timing
            start_time = time.process_time()
            response = retrieval_chain.invoke({
                "input": user_prompt,
                "chat_history": chat_history_str
            })
            response_time = time.process_time() - start_time
            
            # Organize source documents with enhanced metadata
            sources = []
            for doc in response.get('context', []):
                source_info = {
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown Source'),
                    'page': doc.metadata.get('page', 'N/A')
                }
                sources.append(source_info)
            
            return {
                'answer': response['answer'],
                'sources': sources,
                'response_time': response_time
            }
        except Exception as e:
            st.error(f"Error querying documents: {e}")
            return None

    def run(self):
        """
        Streamlit application main runner
        """
        # Page configuration
        st.set_page_config(
            page_title="UMT PRS Intelligent Assistant",
            page_icon="🎓",
            layout="wide"
        )
        
        # Title and Introduction
        st.title("🎓 UMT Participant Relations Section Intelligent Assistant")
        st.markdown("""
        Welcome! I provide precise information from UMT PRS Handbooks. 
        Use the sidebar to manage document embeddings.
        """)
        
        # Sidebar for document embedding management
        st.sidebar.header("Document Embedding Management")
        
        # # Create Embeddings Button
        # if st.sidebar.button("Generate Embeddings"):
        #     with st.spinner("Creating vector embeddings..."):
        #         if self.create_vector_embedding():
        #             st.sidebar.success("Embeddings Generated Successfully!")
        
        # Load Embeddings Button
        if st.sidebar.button("Load Existing Embeddings"):
            with st.spinner("Loading vector embeddings..."):
                if self.load_vector_embedding():
                    st.sidebar.success("Embeddings Loaded Successfully!")
        
        # Initialize user name in session state if not exists
        if "user_name" not in st.session_state:
            st.session_state.user_name = None
        
        # Prompt for user name if not set
        if not st.session_state.user_name:
            st.session_state.user_name = st.text_input("Please enter your name:")
            if st.session_state.user_name:
                st.success(f"Welcome, {st.session_state.user_name}!")
        
        # Chat Interface
        st.header(f"Ask Your PRS Questions, {st.session_state.user_name or 'Student'}")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_prompt = st.chat_input(f"What would you like to know about UMT PRS services, {st.session_state.user_name or 'Student'}?")
        
        if user_prompt:
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": user_prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Get document query response
            response = self.query_documents(user_prompt)
            
            if response:
                # Personalize the response if user name is known
                if st.session_state.user_name:
                    # response_with_name = f"Hello {st.session_state.user_name}{response['answer']}"
                    response_with_name = response['answer']
                else:
                    response_with_name = response['answer']
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_with_name
                })
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(response_with_name)
                
                # Response time and sources
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"Response Time: {response['response_time']:.4f} seconds")
                
                # Expandable source references
                with st.expander("Source Documents"):
                    for i, source in enumerate(response['sources'], 1):
                        st.write(f"Source {i}:")
                        st.write(f"**Document:** {source['source']}")
                        st.write(f"**Page:** {source['page']}")
                        st.write("**Content Excerpt:**")
                        st.code(source['content'][:300] + "...", language="")
                        st.write("---")

def main():
    chatbot = PRSChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()
