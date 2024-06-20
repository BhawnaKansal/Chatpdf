import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
load_dotenv()
 
def main():
    st.header("Chat with PDF")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    


    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # embeddings
        file_name = pdf.name[:-4]
        st.write(f'{file_name}')
        
 
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
 
        # Accept user questions/query
        query = st.text_input("Ask the question:")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(organization='org-G2GcQGiQlXaWGd1iwrEIUWDb',temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()








