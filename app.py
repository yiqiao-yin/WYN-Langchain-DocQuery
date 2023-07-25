import pandas as pd
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def generate_response_from_txt(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever,
        )
        return qa.run(query_text)


# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander("See DataFrame"):
        st.write(df)
    return df


# Generate LLM response
def generate_response_from_csv(csv_file, input_query):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0613", temperature=0.2, openai_api_key=openai_api_key
    )
    df = load_csv(csv_file)
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )
    # Perform Query using the Agent
    response = agent.run(input_query)
    return st.success(response)


# Page title
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

# Tabs
tab1, tab2 = st.tabs(["TXT", "CSV"])

# Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

with tab1:
    # File upload
    uploaded_file = st.file_uploader("Upload an article", type="txt")
    # Query text
    query_text = st.text_input(
        "Enter your question:",
        placeholder="Please provide a short summary.",
        disabled=not uploaded_file,
    )

    # Form input and query
    result = []
    with st.form("myform", clear_on_submit=True):
        # openai_api_key = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     disabled=not (uploaded_file and query_text),
        # )
        submitted = st.form_submit_button(
            "Submit", disabled=not (uploaded_file and query_text)
        )
        if submitted and openai_api_key.startswith("sk-"):
            with st.spinner("Calculating..."):
                response = generate_response_from_txt(
                    uploaded_file, openai_api_key, query_text
                )
                result.append(response)
                del openai_api_key

    if len(result):
        st.info(response)

with tab2:
    # Input widgets
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    question_list = [
        "How many rows are there?",
        "What is the range of values for MolWt with logS greater than 0?",
        "How many rows have MolLogP value greater than 0.",
        "Other",
    ]
    query_text = st.text_input(
        "Enter your question here", "What is the column name?"
    )

    # App logic
    if query_text == "Other":
        query_text = st.text_input(
            "Enter your query:",
            placeholder="Enter query here ...",
            disabled=not uploaded_file,
        )
    # if not openai_api_key.startswith("sk-"):
    #     st.warning("Please enter your OpenAI API key!", icon="⚠")
    if openai_api_key.startswith("sk-") and (uploaded_file is not None):
        st.header("Output")
        generate_response_from_csv(uploaded_file, query_text)
