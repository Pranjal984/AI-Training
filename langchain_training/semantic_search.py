import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()



if __name__ == "__main__":

    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    file_path = "@TCS NQT MCQs @Hiringhustle 05b8be2e33ec40689dbc817afe5f7f25.pdf"
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






