import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDFs")

    st.header("Ask Your PDFs :scroll:")

    #uploading file 
    pdf = st.file_uploader("Upload your Document Here", type="pdf")

    #extracting the file 
    if pdf is not None:
        pdf_reader =PdfReader(pdf)
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spliting text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size =1000,
            chunk_overlap =200,
            length_function =len
        )
        chunks =text_splitter.split_text(text)
        
        # creating embeddings
        embeddings = OpenAIEmbeddings()
        document = faiss.FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your PDFs:")
        if user_question:
            docs = document.similarity_search(user_question)
            
            llm = openai.OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            #with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question = user_question)
                #print(cb)
            st.write(response) 
        
        

        




if __name__ == '__main__':
    main()