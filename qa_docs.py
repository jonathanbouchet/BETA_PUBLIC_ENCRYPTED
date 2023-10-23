import streamlit as st
from firebase_admin import firestore
import os
import logging
from datetime import datetime
from utils import get_tokens, download_transcript, encode_payload
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import time

from models import Tags0
from app import VERSION

log_name = "qa"


def _get_logger(name):
    loglevel = logging.INFO
    l = logging.getLogger(name)
    if not getattr(l, 'handler_set', None):
        print("setting new QA logger")
        l.setLevel(loglevel)
        h = logging.FileHandler(filename=name + ".log")
        f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        h.setFormatter(f)
        l.addHandler(h)
        l.setLevel(loglevel)
        l.handler_set = True
        l.propagate = False
    return l


def get_document(uploaded_files, openai_api_key):
    """
    :param openai_api_key:
    :param uploaded_files:
    :return:
    """
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def new_qa():
    """
    entry point of the QA doc
    :return:
    """

    if not st.session_state["logout"] or st.session_state["authentication_status"] is True:

        user_details = Tags0()

        if 'qa_doc_api_key_set' not in st.session_state:
            st.session_state.qa_doc_api_key_set = False
            st.session_state.qa_doc_uploaded = False
        if 'messages_chat' not in st.session_state:
            st.session_state["messages_chat"] = []
        if 'messages_QA' not in st.session_state:
            st.session_state["messages_QA"] = []
        if 'total_tokens' not in st.session_state:
            st.session_state.total_tokens = 0
        if 'life_insurance_model' not in st.session_state:
            st.session_state.life_insurance_model = user_details

        st.sidebar.markdown(f"""version {VERSION}""")

        if st.sidebar.button("Logout", help="quit session"):
            db = firestore.client()  # log in table
            collection_name = st.secrets["firestore_collection"]
            payload = encode_payload(st.session_state)
            doc_ref = db.collection(f"{collection_name}").document()  # create a new document.ID
            doc_ref.set(payload)  # add obj to collection
            db.close()
            st.empty()  # clear page

            # clean log messages
            for current_log in ["chat_messages.txt", "qa_messages.txt", "chat_messages.pdf", "qa_messages.pdf"]:
                if os.path.exists(current_log):
                    os.remove(current_log)

            st.session_state["logout"] = True
            st.session_state["name"] = None
            st.session_state["username"] = None
            st.session_state["authentication_status"] = None
            st.session_state["login_connection_time"] = None
            st.session_state['messages_chat'] = []
            st.session_state['messages_QA'] = []
            return st.session_state["logout"]

        st.title("Reflexive AI")
        st.header("QA documents")

        if st.sidebar.button("Download transcripts", help="download the chat history with the agent"):
            download_transcript(st.session_state["messages_QA"], log_name)

        model = st.sidebar.selectbox(
            label=":blue[MODEL]",
            options=["gpt-3.5-turbo",
                     "gpt-4"],
            help="openAI model(GPT-4 recommended)")

        show_tokens = st.sidebar.radio(label=":blue[Display tokens]",
                                       options=('Yes', 'No'),
                                       help="show the number of tokens used by the LLM")

        # Set API key if not yet
        openai_api_key = st.sidebar.text_input(
            ":blue[API-KEY]",
            placeholder="Paste your OpenAI API key here",
            type="password",
            help="format is st-***")

        st.sidebar.markdown("How to:")
        st.sidebar.markdown("1. Choose model")
        st.sidebar.markdown("2. Enter your openAI api key")
        st.sidebar.markdown("3. Upload pdf")

        if openai_api_key:

            # openai.api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key

            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = model

            uploaded_files = st.sidebar.file_uploader(
                label="Upload PDF files", type=["pdf"], accept_multiple_files=True
            )
            if not uploaded_files:
                st.info("Please upload PDF documents to continue.")
                st.stop()
            st.session_state.qa_doc_uploaded = True

            tmp_retriever = get_document(uploaded_files, openai_api_key)
            retriever = tmp_retriever.as_retriever()

            # template = """
            # You are a assistant to answer {question} ONLY about the {context} of the data provided.
            # If a question is not about the {context}, reply politely that you cannot answer it.
            # If the user asks a {question} about his identity, reply in a politely manner
            # You also have access to the {chat_history} to reply.
            # Context: {context}
            # -----------------------
            # History: {chat_history}
            # =======================
            # Human: {question}
            # Chatbot:
            # """

            template = """
            You are a helpful assistant able to answer query about the following context {context} only.
            If the query is not about the context {context}, reply politely that you cannot answer it.
            Context: {context}
            -----------------------
            History: {chat_history}
            =======================
            Human: {question}
            Chatbot:"""

            # Create a prompt using the template
            prompt = PromptTemplate(
                input_variables=["chat_history", "context", "question"],
                template=template
            )

            llm = ChatOpenAI(temperature=0,
                             model_name=st.session_state["openai_model"],
                             openai_api_key=openai_api_key)

            # Set up conversation memory
            memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=10,
                input_key="question",
                return_messages=True
            )

            # Set up the retrieval-based conversational AI
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                # verbose=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "memory": memory,
                }
            )

            # initialization of the session_state
            if len(st.session_state["messages_QA"]) == 0:
                # greetings message
                greetings = {
                    "role": "assistant",
                    "content": "What is this document about ?"
                }
                st.session_state["messages_QA"].append(greetings)

            # display chat messages from history on app rerun
            for message in st.session_state.messages_QA:
                # don't print the system content
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # react to user input
            if prompt := st.chat_input("Ask a question about the uploaded file"):
                # display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                    if show_tokens == "Yes":
                        prompt_tokens = get_tokens(prompt, model)
                        st.session_state.total_tokens += prompt_tokens
                        tokens_count = st.empty()
                        tokens_count.caption(f"""query used {prompt_tokens} tokens """)

                st.session_state["messages_QA"].append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    res = qa.run({"query": prompt})
                    for word_response in res.split(" "):
                        full_response += word_response
                        full_response += " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    if show_tokens == "Yes":
                        assistant_tokens = get_tokens(full_response, model)
                        st.session_state.total_tokens += assistant_tokens
                        tokens_count = st.empty()
                        tokens_count.caption(f"""assistant used {assistant_tokens} tokens """)
                # add assistant response to chat history
                st.session_state["messages_QA"].append({"role": "assistant", "content": full_response})


# Run the Streamlit app
if __name__ == "__main__":
    new_qa()
