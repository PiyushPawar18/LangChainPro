import streamlit as st
import pickle
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from apikey import GROQ_API_KEY  # Ensure this file contains the correct API key
from groq import Groq
from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors from scikit-learn

# Streamlit UI Setup
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "vectorstore.pkl"

main_placeholder = st.empty()

# Initialize Groq Client
try:
    client = Groq(api_key=GROQ_API_KEY)  # Ensure compatibility with Groq library
except AttributeError as e:
    st.error(f"Groq client initialization error: {e}. Check the library version.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

if process_url_clicked:
    try:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
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

        # Use NearestNeighbors from scikit-learn instead of FAISS
        index = NearestNeighbors(n_neighbors=5, metric='cosine')
        index.fit(embeddings)
        
        vectorstore = {
            "index": index,
            "texts": texts,
            "metadata": [doc.metadata for doc in docs]
        }

        # Save the vectorstore to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("URLs processed and index created successfully!")
    except Exception as e:
        st.error(f"Error during processing: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                index = vectorstore["index"]
                texts = vectorstore["texts"]
                metadata = vectorstore["metadata"]

                # Retrieve top 5 relevant documents using NearestNeighbors
                query_embedding = model.encode([query])
                distances, indices = index.kneighbors(query_embedding, n_neighbors=5)
                retrieved_docs = [texts[i] for i in indices[0]]

                # Construct context for Groq completion
                context = "\n".join(retrieved_docs)

                # Get answer from Groq
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context: {context}\n\nQuestion: {query}",
                        }
                    ],
                    model="llama3-8b-8192"
                )
                result = chat_completion.choices[0].message.content

                st.header("Answer")
                st.write(result)

                # Display sources
                st.subheader("Sources:")
                for doc in retrieved_docs:
                    st.write(doc)
        else:
            st.warning("Vectorstore file not found. Process URLs first!")
    except Exception as e:
        st.error(f"Error during question answering: {e}")
