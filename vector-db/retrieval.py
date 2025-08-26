import os
from tempfile import template

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import embeddings

load_dotenv()

if __name__ == "__main__":

    print("Retrieving...")

    embeddings=OpenAIEmbeddings()
    llm=ChatOpenAI()

    query= "what is pinecone in machine language?"

    chain=PromptTemplate.from_template(template=query) | llm

   # result=chain.invoke(input={})

   # print(result.content)       # Without RAG, question directly to LLM

    vectorstore=PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX"],
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain=create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain=create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result=retrieval_chain.invoke({"query":query})

    print(result)

