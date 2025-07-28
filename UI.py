import streamlit as st
import pickle
import os
from langchain.document_loaders import UnstructuredURLLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("Article Summary Tool")
st.sidebar.title("Articles URL")

raw_urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Url {i+1}")
    raw_urls.append(url)

urls = [url.strip() for url in raw_urls if url.strip()]

# Embed button
embed_clicked = st.sidebar.button("Load & Embed URLs")
file_path = "vectorstore.pkl"
# Display loading log for embed
if embed_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        with st.spinner("Loading, embedding, and saving to vector database...This may take a few"):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", "."]
            )
            chunks = splitter.split_documents(data)
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embedding=embedding_function)
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.success("Successful, ask your question")
# Question asking UI
main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")
ask_clicked = st.button("Ask question")

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        if os.path.exists(file_path):
            with st.spinner("Retrieving answer...This may take a few"):
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    model_name = "google/flan-t5-small"

                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")  # or .to("cuda") if you have a GPU

                    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, device=0)
                    llm = HuggingFacePipeline(pipeline=pipe)

                    retriever = vectorstore.as_retriever(search_type="similarity", k=4)

                    qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            return_source_documents=True
                    )
                    result = qa_chain({"query": query})
                    st.header("Answer")
                    st.write(result["result"])


                    # Show only clean source URLs
                    source_docs = result.get("source_documents", [])
                    unique_sources = set()

                    for doc in source_docs:
                        source = doc.metadata.get("source")
                        if source:
                            unique_sources.add(source)

                    if unique_sources:
                        st.subheader("Sources:")
                        for source in unique_sources:
                            st.markdown(f"- [{source}]({source})", unsafe_allow_html=True)


