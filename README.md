# 🦜🔗 LangChain - Ask the Doc

Build a Document Question Answering app using LangChain and Streamlit.

# README File

This is a README file for the "Ask the Doc App" project. This project allows users to upload an article and ask questions about the content using OpenAI's language models.

## Installation

To install the necessary dependencies, run the following command:

```
pip install streamlit
```

In addition, make sure you have the required Python packages installed:

- langchain.llms
- langchain.text_splitter
- langchain.embeddings
- langchain.vectorstores

## Usage

To use the "Ask the Doc App", follow these steps:

1. Import the necessary libraries:

   ```python
   import streamlit as st
   from langchain.llms import OpenAI
   from langchain.text_splitter import CharacterTextSplitter
   from langchain.embeddings import OpenAIEmbeddings
   from langchain.vectorstores import Chroma
   from langchain.chains import RetrievalQA
   ```

2. Define the `generate_response` function, which takes in an uploaded file, OpenAI API key, and query text as parameters. This function loads the document, splits it into chunks, creates embeddings, creates a vectorstore, creates a retriever interface, and runs the retrieval QA chain. It returns the response.

3. Set the page configuration and title:

   ```python
   st.set_page_config(page_title='🦜🔗 Ask the Doc App')
   st.title('🦜🔗 Ask the Doc App')
   ```

4. Add a file uploader and a text input for the query:

   ```python
   uploaded_file = st.file_uploader('Upload an article', type='txt')
   query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)
   ```

5. Create a form for user input and query submission:

   ```python
   result = []
   with st.form('myform', clear_on_submit=True):
       openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
       submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
       if submitted and openai_api_key.startswith('sk-'):
           with st.spinner('Calculating...'):
               response = generate_response(uploaded_file, openai_api_key, query_text)
               result.append(response)
               del openai_api_key
   ```

6. Display the result, if available:

   ```python
   if len(result):
       st.info(response)
   ```

## Configuration

The page title is set to "🦜🔗 Ask the Doc App" using `st.set_page_config()`. You can change this title according to your preference.

## Input

- The user can upload an article file (.txt) using the `st.file_uploader()` function.
- The user can enter a question/query in the text input field using the `st.text_input()` function.

## Output

- The app displays the response generated by the `generate_response()` function using `st.info()`.