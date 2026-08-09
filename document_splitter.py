from langchain_text_splitters import RecursiveCharacterTextSplitter


def epub_document_splitter(document_list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # paragraph-first, falls back gracefully
    )
    chunks = splitter.split_documents(document_list)

    return chunks
