from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

from watsonxlangchain import LangChainInterface

creds= {
    'apikey':'ZNPRw05jup0f2SFOCFkAtfZskeCPL6fESGaYAq1EWRp7',
    'url':'https://us-south.ml.cloud.ibm.com' 
}

llm = LangChainInterface(
    credential=creds,
    model='meta-llama/llama-2-70b-chat',
    params= {
        'decoding_method':'sample',
        'max_new_tokens':200,
        'temperature':0.5
    },
    project_id='dd58941e-28cc-4abe-9189-9bebc2f2edec')
@st.cache_resource
def load_pdf():
    pdf_name=''
    loaders=[PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    
    return index

index= load_pdf()

chain= RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question')


st.title('Ask watsonx')

if 'message' not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(['role']).markdown(message['content'])

prompt = st.chat_input("Enter your Prompt Here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response = chain.run(prompt)
    st.chat_message('assistant').markdown(response)
    st.session_state.message.append(
        {'role':'assistant', 'content':response}
    )
