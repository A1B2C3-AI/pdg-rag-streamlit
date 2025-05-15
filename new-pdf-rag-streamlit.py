# app.py - Document Assistant with Self-Learning RAG (Compatible Version)

import streamlit as st
import os
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = r"D:\DIAMOND.pdf"
MODEL_NAME = "qwen3"
EMBEDDING_MODEL = r"D:\Local_model\embeding_model\bge-large-en-v1.5"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
LEARNING_DB_NAME = "learning_db"
LEARNING_DB_DIR = "./learning_chroma_db"

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = PyMuPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        if os.path.exists(PERSIST_DIRECTORY):
            vector_db = Chroma(
                embedding_function=embedding_model,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Loaded existing vector database.")
            return vector_db
        
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        chunks = split_documents(data)
        if not chunks:
            raise ValueError("No document chunks created")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector DB: {str(e)}")
        st.error(f"Failed to load vector database: {str(e)}")
        return None

@st.cache_resource
def load_learning_db():
    """Load or create the learning vector database."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if os.path.exists(LEARNING_DB_DIR):
            learning_db = Chroma(
                embedding_function=embedding_model,
                collection_name=LEARNING_DB_NAME,
                persist_directory=LEARNING_DB_DIR,
            )
            logging.info("Loaded existing learning database.")
            return learning_db
        
        # Create with a dummy document to avoid empty collection
        dummy_doc = Document(
            page_content="Initial document",
            metadata={"source": "system"}
        )
        
        learning_db = Chroma.from_documents(
            documents=[dummy_doc],
            embedding=embedding_model,
            collection_name=LEARNING_DB_NAME,
            persist_directory=LEARNING_DB_DIR,
        )
        learning_db.persist()
        logging.info("Created new learning database with initial document.")
        return learning_db
    except Exception as e:
        logging.error(f"Error loading learning DB: {str(e)}")
        st.error(f"Failed to load learning database: {str(e)}")
        return None

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    try:
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm, 
            prompt=QUERY_PROMPT
        )
        logging.info("Retriever created.")
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {str(e)}")
        return vector_db.as_retriever()

def create_chain(retriever, llm, learning_db):
    """Create the chain with self-learning capability."""
    try:
        template = """Answer the question below based on the provided context and learned knowledge.
Provide only the direct answer without explanations. If you don't know, say "I don't know."

Context: {context}
Learned Knowledge: {learned_knowledge}
Question: {question}

Direct Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def get_learned_knowledge(question):
            try:
                if learning_db:
                    learned_docs = learning_db.similarity_search(question, k=2)
                    if learned_docs:
                        return "\nLearned from previous interactions:\n" + format_docs(learned_docs)
                return ""
            except Exception as e:
                logging.error(f"Error retrieving learned knowledge: {str(e)}")
                return ""
        
        chain = (
            {
                "context": retriever | format_docs,
                "learned_knowledge": RunnablePassthrough() | get_learned_knowledge,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    except Exception as e:
        logging.error(f"Error creating chain: {str(e)}")
        raise

def self_learn(learning_db, question, answer):
    """Store successful Q&A pairs in the learning database."""
    try:
        content = f"Question: {question}\nAnswer: {answer}"
        doc = Document(page_content=content, metadata={"question": question})
        
        learning_db.add_documents([doc])
        learning_db.persist()
        logging.info(f"Learned new Q&A pair: {question[:50]}...")
        return True
    except Exception as e:
        logging.error(f"Error in self-learning: {str(e)}")
        return False

def main():
    st.title("Document Assistant with Self-Learning")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None
    
    try:
        # Load databases
        vector_db = load_vector_db()
        learning_db = load_learning_db()
        
        if vector_db is None or learning_db is None:
            st.error("Failed to initialize databases. Please check logs.")
            return
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return
    
    # Handle exit commands first
    user_input = st.chat_input("Enter your question:")
    
    if user_input:
        if user_input.lower().strip() in ["bye", "exit", "quit"]:
            st.info("Chat ended. Refresh page to start again.")
            return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        try:
            user_doc_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(user_doc_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("PDF uploaded successfully!")
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
    
    # Display chat history (updated to be compatible with older Streamlit versions)
    for user_msg, bot_msg, _ in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
    
    if user_input and user_input.lower().strip() not in ["bye", "exit", "quit"]:
        with st.spinner("Generating response..."):
            try:
                llm = ChatOllama(model=MODEL_NAME, temperature=0)
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm, learning_db)
                
                response = chain.invoke(user_input)
                st.session_state.current_answer = response
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.chat_history.append((user_input, response, None))
                
            except Exception as e:
                st.error(f"An error occurred while generating response: {str(e)}")
                logging.error(f"Response generation error: {str(e)}")
    
    # Add feedback buttons for self-learning
    if st.session_state.current_answer and st.session_state.chat_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Answer is correct"):
                last_question = st.session_state.chat_history[-1][0]
                last_answer = st.session_state.chat_history[-1][1]
                if self_learn(learning_db, last_question, last_answer):
                    st.success("Thanks! I've learned from this interaction.")
                else:
                    st.warning("Learning failed - answer not saved.")
                st.session_state.chat_history[-1] = (
                    st.session_state.chat_history[-1][0],
                    st.session_state.chat_history[-1][1],
                    True
                )
        with col2:
            if st.button("👎 Answer needs improvement"):
                st.info("I'll try to do better next time.")
                st.session_state.chat_history[-1] = (
                    st.session_state.chat_history[-1][0],
                    st.session_state.chat_history[-1][1],
                    False
                )

if __name__ == "__main__":
    main()