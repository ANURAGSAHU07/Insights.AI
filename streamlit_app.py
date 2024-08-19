import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Market Research Analyzer", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #1E90FF;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Retrieve Google API key from environment variable
google_api_key = os.environ.get('GOOGLE_API_KEY')

# google_api_key = "Your api key , if you want to give it directly"

if not google_api_key:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize the LLM
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-exp-0801", 
        temperature=0.9,
        google_api_key=google_api_key  # Use the API key from environment variable
    )

llm = get_llm()


# Title
st.markdown('<p class="big-font">Market Research Analyzer 📈</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Control Panel")

# URL input
st.sidebar.subheader("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

# Process button
process_url_clicked = st.sidebar.button("Process URLs", key="process")

# Main content area
main_placeholder = st.empty()

# Function to process URLs and create embeddings
def process_urls(urls):
    if not urls:
        st.warning("Please enter at least one URL before processing.")
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Data Loading...Started...✅")
    loader = WebBaseLoader(urls)
    data = loader.load()
    progress_bar.progress(25)
    
    status_text.text("Text Splitting...Started...✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    progress_bar.progress(50)
    
    status_text.text("Embedding Vector Building...Started...✅")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_gemni = FAISS.from_documents(docs, embeddings)
    progress_bar.progress(75)
    
    status_text.text("Embedding Vector Building...Completed...✅")
    progress_bar.progress(100)
    
    return vectorstore_gemni

# Process URLs when button is clicked
if process_url_clicked:
    with st.spinner("Processing URLs..."):
        vectorstore_gemni = process_urls(urls)
        if vectorstore_gemni:
            file_path = "Gemniembeddings"
            vectorstore_gemni.save_local(file_path)
            st.success(f"Embedding Vector saved in the location {file_path} ✅")

# Query input and processing
query = st.text_input("Ask a question about the processed articles:", key="query")
if query:
    file_path = "Gemniembeddings/index.faiss"
    if os.path.exists(file_path):
        with st.spinner("Analyzing..."):
            vectorstore = FAISS.load_local("Gemniembeddings", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization = True)
            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            results = chain({"question": query}, return_only_outputs=True)
            
            st.subheader("Answer")
            st.write(results["answer"])

    else:
        st.warning("Please process some URLs before asking questions.")

# Add a feature to compare multiple articles
if st.sidebar.checkbox("Compare Articles"):
    st.subheader("Article Comparison")
    comparison_query = st.text_input("Enter a topic to compare across articles:", key="comparison")
    if comparison_query and os.path.exists("Gemniembeddings/index.faiss"):
        vectorstore = FAISS.load_local("Gemniembeddings", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        
        comparison_results = []
        for url in urls:
            result = chain({"question": f"Summarize the perspective of {url} on {comparison_query}"}, return_only_outputs=True)
            comparison_results.append({"URL": url, "Summary": result["answer"]})
        
        df = pd.DataFrame(comparison_results)
        st.table(df)

# Add a feature to generate a market research report
if st.sidebar.button("Generate Market Research Report"):
    if os.path.exists("Gemniembeddings/index.faiss"):
        st.subheader("Market Research Report")
        report_prompt = "Generate a comprehensive market research report based on the analyzed articles. Include sections on market trends, key players, opportunities, and challenges."
        
        with st.spinner("Generating report..."):
            vectorstore = FAISS.load_local("Gemniembeddings", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            report = chain({"question": report_prompt}, return_only_outputs=True)
            
            st.markdown(report["answer"])
            
            # Option to download the report
            report_text = report["answer"]
            st.download_button(
                label="Download Report",
                data=report_text,
                file_name="market_research_report.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please process some URLs before generating a report.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ by ANURAG")
