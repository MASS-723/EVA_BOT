import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template,user_template
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 


def get_pdf_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size =1000,
        chunk_overlap =200,
        length_function =len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings
    document = faiss.FAISS.from_texts(texts = text_chunks, embedding = embeddings )
    return document

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with you PDFs and DOCs", page_icon=":scroll:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with your PDFS :scroll:")
    user_question = st.text_input("Ask a question about your PDFs")
    if user_question:
        docs = document.similarity_search(user_question)
            
        llm = openai.OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
         #with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question = user_question)
            #print(cb)
        st.write(response)

    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload Your File Here and Click On 'Upload'", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks 
                text_chunks = get_pdf_chunks(raw_text)
                

                # create vector store
                document = get_vectorstore(text_chunks)

                # creating conversation chain
                st.session_state.conversation = get_conversation_chain(document)
    

     



if __name__ == '__main__':
    main()