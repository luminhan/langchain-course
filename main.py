from document_splitter import epub_document_splitter
from langchain_voyageai import VoyageAIEmbeddings
from epub_loader import epub_loader
from vector_database import pinecone_upsert
from dotenv import load_dotenv

load_dotenv()


def process_pinecone_upsert_pipeline():
    documents_list = epub_loader()
    chunks = epub_document_splitter(documents_list)
    embeddings_model = VoyageAIEmbeddings(model="voyage-3.5")  # just the model, not called

    pinecone_upsert(embeddings_model, chunks)


def main():
    process_pinecone_upsert_pipeline()
    print("upserting data to pinecone database done!")


if __name__ == "__main__":
    main()
