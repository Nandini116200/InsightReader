from gtts import gTTS
import os
import streamlit as st
from streamlit.components.v1 import html
import base64
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import streamlit.components.v1 as components
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import re
from collections import Counter

def extract_keywords(text, num_keywords=5):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(["the", "is", "in", "and", "to", "of", "a", "for", "with", "on", "that", "this", "it", "as"])
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    most_common = Counter(keywords).most_common(num_keywords)
    return [word for word, _ in most_common]

st.set_page_config(page_title="AI-Chatbot", layout="wide")



# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME="llama3-70b-8192"


# Extract text from PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Embed text
def embed_pdf_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    doc_chunks = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    st.session_state.vectorstore = vectorstore
    return vectorstore

#This function selects only the top 5 most relevant chunks based on your question and sends those to the LLM.It reduces the token size drastically, and you won’t hit the 5000-token wall.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Select relevant chunks using TF-IDF
def get_relevant_chunks(chunks, query, top_k=5):
    vectorizer = TfidfVectorizer().fit(chunks + [query])
    vectors = vectorizer.transform(chunks + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_indices = similarity.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def extract_text_from_image_pdf(pdf_path):
    # Poppler path set karo
    poppler_path = r"C:\Users\Nandini\Downloads\Release-23.01.0-0\poppler-23.01.0\Library\bin"
    # Tesseract path set karo
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#2b
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        full_text = ""
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i}.png")
            image.save(image_path, "PNG")
            # OCR se text nikalo
            text = pytesseract.image_to_string(Image.open(image_path))
            full_text += text + "\n"
        return full_text

#2c
def is_text_based_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    return True  # At least one page has extractable text
        return False  # No extractable text found
    except:
        return False  # Error in reading, assume image-based


# QA chain
def get_answer(docs, question):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs, question=question)
    return result

# Main logic

def main():

    # Set Streamlit page config
    st.set_page_config(page_title="AI-Chatbot", layout="centered")

    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .chat-container {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .question {
            color: #2980b9;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .answer {
            color: #2c3e50;
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>AI- Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload a PDF and ask questions. Get instant answers with logic-based accuracy!</div>", unsafe_allow_html=True)
    ("gsk_7r0U6OvZbCa9XV@Fb8lVdWGdyb3FYRfCPy!zW6QJ0e2MsRV@!Jyiiqbt")

    st.title("📄Insight Reader")
    st.markdown("---")

    # Initialize message history if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


    if uploaded_file:
        with st.spinner("Reading and indexing your PDF..."):
             # Save uploaded file temporarily to disk
            temp_pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                 f.write(uploaded_file.read())

        # Detect PDF type
        if is_text_based_pdf(temp_pdf_path):
            text = ""
            reader = PdfReader(temp_pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        else:
            text = extract_text_from_image_pdf(temp_pdf_path)

        # ✅ Always embed after extracting text
        vectorstore = embed_pdf_text(text)
        st.success("PDF processed successfully!")

        question = st.text_input("Ask a question about the PDF:")

        if question:
            with st.spinner("Searching relevant content..."):
                if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
                   retriever = st.session_state.vectorstore.as_retriever() 
                else: 
                   st.warning("Please upload a PDF first.")
                   return
            
                docs = retriever.get_relevant_documents(question)

               # Step 2.4 - keyword filtering
            keywords = extract_keywords(question)
            filtered_chunks = [doc for doc in docs if any(k in doc.page_content.lower() for k in keywords)]

            if not filtered_chunks:
                 st.warning("No matching content found based on keywords. Using default results...")
                 filtered_chunks = docs  # fallback

            with st.spinner("Generating answer..."):
                  answer = get_answer(filtered_chunks, question)
                  tts = gTTS(text=answer, lang='en')  # ya lang='hi' if needed
                  tts.save("answer.mp3")
                  audio_file = open("answer.mp3", "rb")
                  st.audio(audio_file, format="audio/mp3")
                  audio_file.close()
                  os.remove("answer.mp3")
                
            # Prepend latest Q&A to top
            st.session_state.chat_history.append({"question": question, "answer": answer})

        # Display Q&A history with latest on top
        for i in range(len(st.session_state['chat_history'])-1, -1, -1):
            q = st.session_state['chat_history'][i]["question"]
            a = st.session_state['chat_history'][i]["answer"]
            st.markdown(f"""
            <div class='chat-container'>
                    <div class='question'>🧑‍💼 You: {q}</div>
                    <div class='answer'>🤖 Bot: {a}</div>
            </div>
        """,unsafe_allow_html=True)


main()
