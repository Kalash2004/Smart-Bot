## In this file we will implement RAG
from langchain_community.document_loaders import PyPDFLoader # For loading the document 
from langchain_chroma import Chroma  # vector database
from langchain_text_splitters import RecursiveCharacterTextSplitter # splitting the document into text
from langchain_cohere import CohereEmbeddings # converting those text into vector representation
from langchain_cohere import ChatCohere # genrating result 
from langchain_core.runnables import RunnablePassthrough # Pass data forward without change
from langchain_core.output_parsers import StrOutputParser # Get clean, usable text output

    
# We want the combine the output into a single sentence
def format_docs(docs):
    return'\n'.join(doc.page_content for doc in docs)

def rag_chain_pipeline():
    # First we will laod the document which we like to load 
    loader  = PyPDFLoader(file_path=r"C:\Users\Kalash Srivastava\RAG\Research_Paper_On_Artificial_Intelligence_And_Its.pdf")
    docs = loader.load()

    # Now we will split the data into smaller chunks and store in vector database (Chroma)
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
    split = text_splitter.split_documents(docs)
    vectore_db = Chroma.from_documents(documents=split,embedding=CohereEmbeddings(model="embed-english-v3.0",cohere_api_key="dAnko2bn2OA0app21L6Ahxg9tT58l21sTeE4gH4l"))

    # we want the best output so we use as _reriever() to find the best match
    retriver = vectore_db.as_retriever()

    #We are using the predefined prompt to interact with the LLM
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")

    # Now we want to intialize the LLM chat model we will use like for Cohere 
    llm = ChatCohere(cohere_api_key="dAnko2bn2OA0app21L6Ahxg9tT58l21sTeE4gH4l")


    rag_chain = ({'context':retriver|format_docs,'question':RunnablePassthrough()}
                 | prompt
                 | llm
                 | StrOutputParser()
                 )
    return rag_chain
rag_chain = rag_chain_pipeline()




def ask_query(question):
    return rag_chain.invoke(question)







