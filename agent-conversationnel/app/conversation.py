

import os
from langchain import ElasticVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
import lancedb
from langchain.vectorstores import LanceDB
from langchain import PromptTemplate
elasticsearch_uri =os.environ.get('SERVER_URI_ELASTICSEARCH')
token_key=os.environ.get('TOKEN_OPEN_KEY')
def create_conversation() -> ConversationalRetrievalChain:

    embeddings = OpenAIEmbeddings(
        openai_api_key=token_key
    )
    template = """Utilisez les éléments de contexte suivants pour répondre à la question finale. Si vous ne connaissez pas la réponse, dites simplement :
    "
    Je suis désolé, mais nous ne disposons pas de documents contenant la réponse à votre question actuelle. Veuillez essayer de reformuler votre question pour qu'elle soit plus claire.
    Si vous le souhaitez, vous pouvez également obtenir plus d'informations en visitant les sites officiels de la ville de Paris :
    Site de la ville de Paris : https://www.paris.fr/
    Plateforme de données ouvertes de Paris : https://opendata.paris.fr/pages/home/
    Projet PluBio Climatique de Paris : https://plubioclimatique.paris.fr/projet/"
    , n'essayez pas d'inventer une réponse Sauf pour les expressions de politesse (comme merci ...)  et de salutation (comme bonjour ..).

    {context}
    Question: {question}
    Réponse:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    db=ElasticVectorSearch(elasticsearch_url=elasticsearch_uri,index_name='chatbot',embedding=embeddings)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )
 
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=token_key), chain_type='stuff',retriever=db.as_retriever(), memory=memory,chain_type_kwargs=chain_type_kwargs )
    return qa

