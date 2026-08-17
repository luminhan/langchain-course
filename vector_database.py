from langchain_pinecone.vectorstores import PineconeVectorStore


def pinecone_upsert(embeddings_model, texts):
    index_name = "golden-son"
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings_model,
        index_name=index_name
    )
