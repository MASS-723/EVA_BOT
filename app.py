import streamlit as st
from streamlit_option_menu import option_menu
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


st.set_page_config(page_title="Query With Your Own Data ",
                   page_icon=":scroll:", layout="wide")


def get_pdf_text(pdf):
    text = ""
    for pdf in pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorestore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)
    return vectorstore


# Define the get_conversation_chain function with user_question as an argument
def get_conversation_chain(user_question, vectorstore):
    if user_question:
        docs = vectorstore.similarity_search(user_question)
        llm = openai.OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.write(response)


def main():
    load_dotenv()
    with st.sidebar:
        st.markdown("<h1 style='text-align: center'>👾Eva</h1>",
                    unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Chat", "Upload File", "Contacts"],
            icons=['chat-right-dots-fill',
                   'cloud-upload-fill', 'person-rolodex'],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )

    user_question = None  # Initialize user_question outside of if statement

    if selected == 'Chat':
        st.file_uploader(
            "Upload question as an image if you want!", type="png")
        user_question = st.text_input(
            "Ask a question about your document :scroll:", placeholder="Write Something...")

    if selected == 'Upload File':
        st.header("Query With Your Data :scroll:")
        pdf = st.file_uploader(
            "Upload Your Data Here :scroll:", type="pdf", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorestore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    user_question, vectorstore)
                st.write("Successfully Upload, Now Go To Chat And Query Your Data")


if __name__ == '__main__':
    main()
