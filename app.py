import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

#import os





def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator ="\n",
        chunk_size = 1000,
        chunk_overlap =200,
        length_function =len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="hkunlp/instructor-xl", model_kwargs={"temperature": 0.5, "max_lenth": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm =llm,
        retriever =vectorstore.as_retriever(),
        memory= memory 
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("Hello bot", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("Hello bot", message.content), unsafe_allow_html=True)
            
    




def main():
    load_dotenv()
    #print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    st.set_page_config(page_title="Chat with us with your won PDFs and DOCs", page_icon=":scroll:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None


    st.header("chat with your PDFs :scroll:")
    user_question = st.text_input("Ask a question about your document:")



    if user_question:
        handle_userinput(user_question)

    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("your document")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'upload'", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get the pdf text 
                raw_text = get_pdf_text(pdf_docs)
                

                 # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                

                # create vector store
                vectorstore = get_vectorestore(text_chunks)

                # create conversation 
                st.session_state.conversation = get_conversation_chain(vectorstore)
   




if __name__=='__main__':
    main()
  




