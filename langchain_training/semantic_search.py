import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain



load_dotenv()



if __name__ == "__main__":

    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    file_path = "nyse-nke-2023-10K-231100085.pdf"
    loader = PyPDFLoader(file_path)
    pdf_docs = loader.load()
    print(f"\n Loaded {len(pdf_docs)} pages from PDF.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(pdf_docs)
    print(len(all_splits))


    # for i, doc in enumerate(pdf_docs, start=1):
    #     print(f"--- Page {i} ---")
        # print(f"{docs[0].page_content[:200]}\n")
        # print(docs[0].metadata)



    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # vector_1 = embeddings.embed_query(all_splits[0].page_content)
    # vector_2 = embeddings.embed_query(all_splits[1].page_content)

    # assert len(vector_1) == len(vector_2)
    # print(f"Generated vectors of length {len(vector_1)}\n")
    # print(vector_1[:10])



    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )

    print(results[0])



    results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(doc)









