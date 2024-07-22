import streamlit as st
from langchain.llms import GooglePalm
import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings 
from langchain.text_splitter import CharacterTextSplitter  
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain


def main():

    api_key="your API key"
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

# upload file
    input_file= st.file_uploader("Upload your PDF", type="pdf")

# extract the text
    if input_file is not None:
        pdf_reader = PdfReader(input_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()         
         
# split into chunks
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
        chunks = text_splitter.split_text(text)      
        
    # create embeddings
        embeddings =HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")   
        knowledge_base = FAISS.from_texts(chunks, embeddings)        

    # show user input
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:

            docs = knowledge_base.similarity_search(user_question)            
            
            llm = GooglePalm(google_api_key=api_key, temperature=0.1)   

            template = """
    Given the following context, please answer the question concisely.
    If the answer is not found in the context, kindly state "please contact business directly" Don't try to make up an answer.

    Context: {summaries}
    Question: {question}
    """
            prompt = PromptTemplate(template=template, input_variables=["summaries", "question"])

            chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt,document_variable_name="summaries")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.markdown(
        f"<h1 style='color:#ff6666; font-size: 24px;'>Result:</h1>",
        unsafe_allow_html=True,)  
                
            st.write(response)    


if __name__ == '__main__':
    main()        