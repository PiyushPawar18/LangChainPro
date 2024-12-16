import streamlit as st
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from apikey import GROQ_API_KEY
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings

# Download NLTK data
nltk.download('punkt')
warnings.filterwarnings("ignore", category=SyntaxWarning)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Initialize Groq Client
try:
    client = Groq(api_key=GROQ_API_KEY)
except AttributeError as e:
    st.error(f"Groq client initialization error: {e}. Check the library version.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

if process_url_clicked:
    try:
        # Validate URLs
        valid_urls = [url for url in urls if url.startswith("http")]
        if not valid_urls:
            st.error("Please provide valid URLs that start with http or https.")
            st.stop()

        # Load data from URLs
        loader = UnstructuredURLLoader(urls=valid_urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        texts = [doc.page_content for doc in docs]

        # Create embeddings using SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder="models")
        embeddings = model.encode(texts)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")

        # Save embeddings and texts
        vectorstore = {
            "embeddings": embeddings,
            "texts": texts,
            "metadata": [doc.metadata for doc in docs]
        }
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("URLs processed and embeddings created successfully!")
    except Exception as e:
        st.error(f"Error during processing: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    try:
        if os.path.exists("vectorstore.pkl"):
            with open("vectorstore.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                embeddings = vectorstore["embeddings"]
                texts = vectorstore["texts"]

                # Get query embedding
                query_embedding = model.encode([query])
                similarities = cosine_similarity(query_embedding, embeddings)
                top_indices = similarities.argsort()[0][-5:][::-1]

                # Retrieve top documents
                retrieved_docs = [texts[i] for i in top_indices]
                context = "\n".join(retrieved_docs)

                # Get answer from Groq
                chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}"
                    }],
                    model="llama3-8b-8192"
                )
                result = chat_completion.choices[0].message.content

                st.header("Answer")
                st.write(result)
                st.subheader("Sources:")
                for doc in retrieved_docs:
                    st.write(doc)
        else:
            st.warning("Data file not found. Process URLs first!")
    except Exception as e:
        st.error(f"Error during question answering: {e}")
