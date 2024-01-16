from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS
import chromadb
from chromadb.config import Settings
persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts = text_splitter.split_documents(documents)
                #create embeddings
                embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
                #create vector store
                print(texts, embeddings)
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
                db.persist()
                db=None

if __name__ == "__main__":
    main()
    # embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    # db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    # query = "What is parasite drag"
    # docs = db.similarity_search(query)    

    # print(docs[0].page_content)