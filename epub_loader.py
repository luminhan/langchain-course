import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def epub_loader():
    book = epub.read_epub("Red_Rising.epub")

    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            title = soup.find(["h1", "h2"]).get_text() if soup.find(["h1", "h2"]) else item.get_name()
            chapters.append(
                {"title": title,
                 "text": text})

    documents = [
        Document(
            page_content=chapter["text"],
            metadata={"chapter_number": chapter["title"], "chapter_index": i, "source": "Red_Rising.epub"}
        )
        for i, chapter in enumerate(chapters)
    ]

    return documents

