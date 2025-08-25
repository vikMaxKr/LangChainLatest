import os

from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from openai import embeddings

if __name__ == "__main__":
    print("Ingesting...")
    loader=TextLoader("/Users/vikashkumar/AI/LangChainLatest/vector-db/mediumblog.txt")
    document=loader.load()

    print("splitting...")

    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts=text_splitter.split_documents(document)
    print(f"split into {len(texts)} chunks of text")

    embeddings=OpenAIEmbeddings(open_api_key=os.environ.get("OPENAI_API"))

    print("ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("PINECONE_INDEX"))