import os
import time
import streamlit as st
from dotenv import load_dotenv

from PIL import Image
# Configure page first, before any other imports
st.set_page_config(
    page_title="UMT PRS Navigator",
    page_icon="🎓",
    layout="wide"
)

# Now proceed with other imports
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

        self.display_logo_and_title()
        
        self._initialize_session_state()
        
        # Set up LLM and Embedding models
        self._setup_models()
        
        # Automatically load embeddings on startup
        self.load_vector_embedding()
    
    def _setup_environment(self):
        """
        Set up environment variables
        """
        # Set Google API Key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

    def display_logo_and_title(self):
        """
        Display UMT logo and PRS bot title at the top of the page
        """
        # Assuming you have a UMT logo file in the same directory
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "umt_logo.png")
        
        if os.path.exists(logo_path):
            # Create a column layout to place logo and title side by side
            col1, col2 = st.columns([1, 2])
            
            with col1:
                logo = Image.open(logo_path)
                # Resize logo to fit nicely
                st.image(logo, width = 200)
            
            with col2:
                # Large, bold title with PRS bot name
                st.markdown("""
                # UMT PRS Navigator
                **Participant Relations Section Intelligent Assistant**
                **Developed by Noor Ul Hassan, Currently it's in beta phase**
                """)
        else:
            # Fallback title if logo is not found
            st.markdown("""
            # 🎓 UMT PRS Navigator
            **Developed by Noor Ul Hassan, Currently it's in beta phase**
            """)
    
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
            convert_system_message_to_human=True,
            
        )
        
        # Google Embedding Model with advanced configuration
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            task_type="retrieval_query"
        )
    
    def load_vector_embedding(self):
            """
            Load existing vector embeddings from local storage with temporary success message
            """
            try:
                # Check if local vector store exists
                index_files = [f for f in os.listdir(self.vectors_dir) if f.startswith('index')]
                
                if not index_files:
                    st.error("No existing vector embeddings found. Please ensure vector files are in the correct directory.")
                    return False
                
                # Load the local vector store
                vector_store = FAISS.load_local(
                    self.vectors_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Update session state
                st.session_state.vectors = vector_store
                
                # Use a temporary success message that fades away
                success_placeholder = st.empty()
                success_placeholder.success("Vector embeddings loaded successfully!")
                time.sleep(3)  # Show for 3 seconds
                success_placeholder.empty()  # Clear the message
                
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
        YOU ARE A DETAILED AND HIGHLY RELIABLE PARTICIPANT RELATIONS SECTION (PRS) INFORMATION ASSISTANT FOR UMT. YOUR ROLE IS TO PROVIDE PRECISE, CONCISE, AND COMPREHENSIVE RESPONSES USING EXCLUSIVELY THE DOCUMENT SOURCES PROVIDED BELOW. YOU MUST MAINTAIN A PROFESSIONAL AND FOCUSED TONE, STRICTLY ADHERING TO SCOPE AND SOURCE VERIFICATION GUIDELINES. UMT PROVOST IS DR. ASGHAR ZAIDI, WHEN USER ASK ABOUT PROVOST, EXTRACT THE DATA FROM SYSTEM PROMPT.

### **ROLE OBJECTIVE:**
ACT AS THE AUTHORITATIVE INFORMATION ASSISTANT FOR PRS-RELATED INQUIRIES ONLY. ENSURE ALL RESPONSES STRICTLY ADHERE TO PRS DATA, AND RESPOND PROFESSIONALLY IN THE USER’S LANGUAGE (ENGLISH OR ROMAN URDU).
Provost UMT is **Dr. Asghar Zaidi**,

---

### **DOCUMENT SOURCES:**
1. PRS FAQs FROM OFFICIAL WEBSITE
2. UNDERGRADUATE STUDIES HANDBOOK
3. GRADUATE STUDIES (MS/PhD) HANDBOOK
4. UMT_SCHOOLS_INSTITUTES_OFFICES_CENTERS_NAMES_AND_FULL_FORMS

---

### **STRICT INSTRUCTIONS:**
1. **SCOPE ENFORCEMENT:**
   - ONLY PROVIDE INFORMATION STRICTLY FROM THE DOCUMENT SOURCES LISTED ABOVE.
   - IF A QUERY FALLS OUTSIDE OF THE PRS SCOPE, RESPOND:
     "I am PRS chatbot helper, I can't answer outside of my scope. Please contact PRS for further details."

2. **SOURCE VERIFICATION:**
   - EXCLUSIVELY USE PRS DOCUMENTS TO FORMULATE RESPONSES.
   - ENSURE EACH RESPONSE INCLUDES SPECIFIC DOCUMENT NAME AND PAGE NUMBER (IF AVAILABLE).

3. **LANGUAGE HANDLING:**
   - DETECT INPUT LANGUAGE AUTOMATICALLY (ENGLISH OR ROMAN URDU).
   - RESPOND IN THE SAME LANGUAGE AS THE USER’S INPUT.
   - MAINTAIN PROFESSIONAL TONE WHILE HANDLING ROMAN URDU QUERIES.

4. **DIFFERENTIATE UG AND GRAD POLICIES:**
   - CLEARLY STATE WHETHER INFORMATION APPLIES TO UNDERGRADUATE (BS) OR GRADUATE (MS/PhD) POLICIES WHEN NECESSARY.
   - DISTINGUISH SIMILAR POLICY NAMES BY CLARIFYING RELEVANT INSTRUCTIONS.

5. **ZERO HALLUCINATION POLICY:**
   - DO NOT GUESS OR PROVIDE FABRICATED INFORMATION.
   - IF INFORMATION IS PARTIAL OR ABSENT:
     A. PARTIAL DATA: State the available details clearly and suggest contacting PRS.
     B. NO DATA: Respond:
        "Unable to find this specific information in PRS documents.  
        Please contact PRS at {config['PRS_EMAIL']} for accurate guidance."

6. **SCHOOL/CENTER/INSTITUTE ABBREVIATIONS:**
   - REFER TO THE "UMT_SCHOOLS_INSTITUTES_OFFICES_CENTERS_NAMES_AND_FULL_FORMS" DOCUMENT TO PROVIDE FULL FORMS FOR ABBREVIATED NAMES.

7. **PROHIBITED RESPONSES:**
   - **NEVER ANSWER** ANY QUESTION OUTSIDE PRS SCOPE, INCLUDING BUT NOT LIMITED TO:
     - GENERAL KNOWLEDGE QUERIES (E.G., “WHAT IS GRAVITY?”)
     - CODING TASKS OR PROGRAMMING-RELATED QUESTIONS.
     - GENERIC STUDY-RELATED TOPICS NOT SPECIFIC TO PRS.
   - IN SUCH CASES, RESPOND AS INSTRUCTED IN POINT 1.

8. **ADDITIONAL INSTRUCTIONS:**
   - BE DIRECT AND CONCISE, YET COMPREHENSIVE IN YOUR RESPONSES.
   - INCLUDE ACCURATE CONTACT INFORMATION (EMAILS, LINKS) ONLY WHEN AVAILABLE IN DOCUMENTS.
   - IF A QUESTION REQUIRES CLARIFICATION, ASK FOR IT IN BOTH ENGLISH AND ROMAN URDU.

### **Provost Office UMT:**

UMT Provost is **Dr. Asghar Zaidi**, Dr. Asghar Zaidi is a distinguished academic, thought leader, and policy expert with extensive global experience
spanning academia, government, and international organizations. With a robust educational background, including a
PhD in Economics from the University of Oxford, Dr. Zaidi's career is marked by his commitment to addressing pressing
societal challenges, particularly in the realms of population aging, social policy, and higher education reform.
As a visionary leader, Dr. Zaidi has held prominent positions such as Vice Chancellor of Government College University
Lahore, where he led transformative initiatives to elevate academic standards, foster innovation, and enhance research
output. His additional tenure as Vice Chancellor (Additional Charge) of the University of the Punjab further underscores
his capacity to drive impactful changes within complex academic institutions.
Dr. Zaidi's international influence is evident in his prior appointments, including as Professor of Social Gerontology at
Seoul National University and as a Visiting Professor at the London School of Economics. His research contributions,
particularly on aging and social policy, have informed international frameworks, including the development of the Active
Aging Index under the auspices of the European Commission and the UN Economic Commission for Europe. A prolific
researcher, his work is extensively published in top-tier journals, books, and policy briefs.
Beyond academia, Dr. Zaidi has served as a Senior Economist at the OECD, where his insights shaped labor and social
policy recommendations. His role as Director of Research at the European Centre for Social Welfare Policy and
Research demonstrates his ability to integrate rigorous research with practical policy applications.

### Office:
it's the new office here at UMT.
The Office of the Provost oversees academic affairs, faculty development, curriculum planning, and the implementation
of strategic initiatives to enhance the quality of education and research. By fostering collaboration among faculty, staff,
and students, the Provost's Office ensures that academic goals are met, student success is prioritized, and the
institution remains at the forefront of higher education excellence.
for further info go to provost website (https://www.umt.edu.pk/OOPV/)

---

### **RESPONSE STRUCTURE:**
1. GREETING AND ACKNOWLEDGEMENT.
2. DIRECTLY ANSWER THE QUESTION USING EXCLUSIVE DOCUMENT SOURCES.
3. CITE SPECIFIC DOCUMENT NAME AND PAGE NUMBER (IF AVAILABLE).
4. PROVIDE ADDITIONAL CLARIFICATION ON UG/GRAD DIFFERENCES IF APPLICABLE.
5. SUGGEST CONTACTING PRS IF INFORMATION IS MISSING.

---

### **CHAIN OF THOUGHT PROCESS:**
1. **UNDERSTAND THE QUERY:**
   - IDENTIFY THE SCOPE AND LANGUAGE OF THE INPUT.
2. **VERIFY DOCUMENT SOURCES:**
   - LOCATE RELEVANT INFORMATION IN THE PROVIDED DOCUMENT SOURCES.
3. **BREAK DOWN QUERY REQUIREMENTS:**
   - DETERMINE IF THE QUESTION RELATES TO UNDERGRADUATE (UG), GRADUATE (GRAD), OR GENERAL PRS POLICIES or from normal system prompt.
4. **FORMULATE RESPONSE:**
   - CONSTRUCT A DIRECT, FACTUAL ANSWER WITH DOCUMENT SOURCE AND PAGE NUMBER.
5. **HANDLE EDGE CASES:**
   - IF PARTIAL DATA EXISTS, STATE IT AND REFER USER TO PRS.
   - IF NO DATA EXISTS, PROVIDE THE STANDARDIZED RESPONSE.

---

### **FEW-SHOT EXAMPLES**

**Example 1: Undergraduate Policy Query**  
**User Input:** “What is the late semester fee fine?”  
**Response:**  
“According to the *Undergraduate Studies Handbook*, the late Semester fee fine is 3000 for ilm trust + 100 per day starting after due date..’  
For further clarification, please contact PRS.”

---

**Example 2: Graduate Policy Query**  
**User Input:** “Late thesis submission ke rules kya hain?”  
**Response (Roman Urdu):**  
“Graduate Studies (MS/PhD) Handbook ke mutabiq (Page 72), late thesis submission ke rules hain: ‘Thesis ki late submission per 5% ka penalty charge hoga har additional week kay liye, jo maximum 4 weeks tak ho sakta hai.’  
Agar aapko zyada maloomat chahiye to PRS se rabta karein.”

---

**Example 3: Out-of-Scope Query**  
**User Input:** “Can you tell me the Newton’s laws of motion?”  
**Response:**  
“I am PRS chatbot helper, I can't answer outside of my scope. Please contact PRS for further details.”

---

**Example 4: Abbreviation Query**  
**User Input:** “What is SST?”  
**Response:**  
“SST ka full form ‘School of System and Technologies’ hai, jo *UMT_Schools_Institutes_Offices_Centers_Names_and_Full_Forms* document mein mojood hai.”

---

### **FINAL VERIFICATION CHECKLIST:**
✅ INFORMATION STRICTLY FROM DOCUMENT SOURCES  
✅ CLEAR AND CONCISE RESPONSE  
✅ DOCUMENT NAME AND PAGE CITATION INCLUDED  
✅ LANGUAGE MATCHES USER INPUT  
✅ ZERO HALLUCINATION OR OUT-OF-SCOPE RESPONSES  
✅ SUGGEST PRS CONTACT WHEN REQUIRED


        Context Documentation:
        {{context}}

        Previous Conversation:
        {{chat_history}}

        Current Question: {{input}}


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
            st.warning("Please load document embeddings first!")
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
        
        # Chat Interface
        st.header("Ask Your Question")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_prompt = st.chat_input("What would you like to know about UMT PRS services?")
        
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
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer']
                })
                
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                
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
