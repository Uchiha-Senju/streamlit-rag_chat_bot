import streamlit as st
import ollama as olm
import chromadb as cdb
from stqdm import stqdm as tqdm

from typing import List, Optional, Union
from langchain_core.documents.base import Document

from io import BytesIO
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

class CustomPDFLoader(BaseLoader):
  def __init__(self, stream: BytesIO, password: Optional[Union[str, bytes]] = None, extract_images: bool = False):
    self.stream = stream
    self.parser = PyPDFParser(password=password, extract_images=extract_images)

  def load(self) -> List[Document]:
    return list(self.parser.parse(Blob.from_data(self.stream.read())))

def add_pdf_file_to_collection_from_blob(blob_list:list, collection, embed_model:str, char_stride:int = 500, char_overlap:int = 50) :
  for f in tqdm(blob_list) :
    pdf_doc = CustomPDFLoader(f)
    data = pdf_doc.load()
    # split the doc into snippets
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=char_stride, chunk_overlap=char_overlap)
    snippets = text_splitter.split_documents(data)
    
    # Add to collection
    for i,s in enumerate(tqdm(snippets, leave=False)) :
      text = s.page_content
      em = olm.embeddings(
        model=embed_model,
        prompt=text,
      )['embedding']
      s.metadata['source'] = f.name
      collection.add(
        embeddings=[em],
        documents=[text],
        metadatas=[s.metadata],
        ids= [f.name + ' ' + str(i)],
      )

def answer_with_context_and_history(query:str, collection, answering_model:str, embed_model:str, history:str='', top_k:int=20) :
  olm.pull(model=answering_model)
  print("pulled answering model")

  historic_query = olm.generate(
    model = answering_model,
    prompt = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, "
      "just reformulate it if needed and otherwise return it as is.\n"
    ) + f"History : \n {history} \n" + f"Query : {query}"
  )
  historic_query = historic_query['response']
  print(f"query with history : {historic_query}")

  q_embed = olm.embeddings(
    model=embed_model,
    prompt=historic_query
  )['embedding']

  print("querying knowledge base... ", end='')

  context = collection.query(
    query_embeddings=[q_embed],
    n_results = top_k,
    # include=["documents", "distances"]
  )
  # return context
  context_ids = context['ids'][0]
  context_str = context['documents'][0]
  context_dis = context['distances'][0]
  relevant_context = [x for x, dis in zip(context_str,context_dis) if dis < 0.5]
  all_context = '\n\n'.join(relevant_context)

  print("query complete")
  print("Generating response... ", end='')
  answer = olm.generate(
    model = answering_model,
    prompt = (
      f"Using the given context : \n {all_context} \n\n "
      # f"Chat History : {history} \n\n"
      f"Answer the following question. \n"
      f"Question : {historic_query} \n"
      "Answer :"
    )
  )

  print("Answer generated")

  return query, answer['response'], relevant_context

client = cdb.EphemeralClient()
collection = client.get_or_create_collection(name="my_collection",metadata={"hnsw:space": "cosine"})

answering_model = "gemma2:2b"
embed_model = "nomic-embed-text"

st.set_page_config(layout="wide")#, initial_sidebar_state="expanded")
st.header("PDF QA RAG Bot")

css = """
div[data-testid="stFileUploaderDeleteBtn"] {
  display: none;
  visibility: hidden;
}
"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

if 'message_history' not in st.session_state :
  st.session_state.message_history = ''
if 'processed_docs' not in st.session_state :
  st.session_state.processed_docs = []

col1, col2 = st.columns((1,2))

with col1 :
  with st.expander("Input the Knowledge base"):
    documents = st.file_uploader(
      label="File Input",
      type='pdf', 
      accept_multiple_files=True,
    )
    # def process_uploaded_file() :
    if len(documents) != 0 :
      un_processed_docs = [x for x in documents if x.name not in st.session_state.processed_docs]
      with st.spinner("Embedding Docs") :
        add_pdf_file_to_collection_from_blob(un_processed_docs, collection, embed_model)
        print("Processed", un_processed_docs)
      for f in un_processed_docs :
        st.session_state.processed_docs.append(f.name)

# I hate Steamlit
# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# if "i" not in st.session_state :
#   st.session_state.i = 0

with col2 :
  # Display chat messages from history on app rerun
  chat_box = st.container(border=True)
  with chat_box :
    for message in st.session_state.messages :
      with st.chat_message(message["role"]):
        st.markdown(message["content"])
  prompt = st.chat_input("Ask anything")

  # def process_chat_input() :
  with chat_box :
    if prompt :
      with st.chat_message("user") :
        st.markdown(prompt)
        st.session_state.message_history += f"User : {prompt}\n"
        st.session_state.messages.append({"role": "user", "content": prompt})

      with st.spinner("Generating response") :
        query, response, context = answer_with_context_and_history(prompt, collection, answering_model, embed_model, history=st.session_state.message_history)
        # response = "Ret %d" % st.session_state.i
        # st.session_state.i += 1

      with st.chat_message("assistant") :
        st.markdown(response)
        st.session_state.message_history += f"Assistant : {response}\n"
        st.session_state.messages.append({"role": "assistant", "content": response})
  
  # print("#"*5, st.session_state.message_history, "#"*5, sep='\n')
  # print(context)
  
  


  