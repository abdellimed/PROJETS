import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import lancedb
from langchain.vectorstores import LanceDB

from langchain.document_loaders import DirectoryLoader, TextLoader,PyPDFLoader
#from PyPDF2 import PdfReader
from langchain import ElasticVectorSearch



def create_index(file_path: str):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    text_splitter =SpacyTextSplitter(chunk_size=1500)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(
      openai_api_key='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'
    )
    db = ElasticVectorSearch.from_documents(
    texts, embeddings, elasticsearch_url="http://10.185.33.168:9200",index_name="chatbot")



#create_index("./documents_upload/RP_CHOIX.pdf")
#create_index("./documents_upload/RP_DIAGNOSTIC.pdf")
#create_index("./documents_upload/RP_EVAL.pdf")
create_index("./documents_upload/RP_EIE.pdf")
#create_index("./documents_upload/RP_INDIC.pdf")
#create_index("./documents_upload/RP_PREAMBULE.pdf")
create_index("./documents_upload/RP_RNT.pdf")
