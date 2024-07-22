# Langchain Ask PDF (Tutorial)

Demo Video [Video](https://drive.google.com/file/d/1mt_RZ3Xkp4I2jR4powk3sYBQUK9DYz5h/view?usp=sharing).

This is a Python application that allows you to load a PDF and ask questions about it using natural language. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

Ask a question,it will take some time to give answer based on the PDF File Size.



