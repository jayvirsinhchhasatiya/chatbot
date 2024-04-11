import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Replicate
from langchain_community.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile
from yt import get_yt_summary_test
from video import get_video_text


load_dotenv()

# session for documents
def initialize_session_state_document():
    if 'history_document' not in st.session_state:
        st.session_state['history_document'] = []

    if 'generated_document' not in st.session_state:
        st.session_state['generated_document'] = ["Hello! Ask me about Documents🤗"]

    if 'past_document' not in st.session_state:
        st.session_state['past_document'] = ["Hey! 👋"]

# session for youtube video
def initialize_session_state_youtube():
    if 'history_youtube' not in st.session_state:
        st.session_state['history_youtube'] = []

    if 'generated_youtube' not in st.session_state:
        st.session_state['generated_youtube'] = ["Hello! Ask me about YouTube Video🤗"]

    if 'past_youtube' not in st.session_state:
        st.session_state['past_youtube'] = ["Hey! 👋"]

# session for local video
def initialize_session_state_video():
    if 'history_video' not in st.session_state:
        st.session_state['history_video'] = []

    if 'generated_video' not in st.session_state:
        st.session_state['generated_video'] = ["Hello! Ask me about Video🤗"]

    if 'past_video' not in st.session_state:
        st.session_state['past_video'] = ["Hey! 👋"]

# session for website
def initialize_session_state_website():
    if 'history_website' not in st.session_state:
        st.session_state['history_website'] = []

    if 'generated_website' not in st.session_state:
        st.session_state['generated_website'] = ["Hello! Ask me about Website🤗"]

    if 'past_website' not in st.session_state:
        st.session_state['past_website'] = ["Hey! 👋"]

        
# clear and initialize session state
def clear_and_initialize_session_state(option):
    if option == "Document ChatBot":
        initialize_session_state_document()
    elif option == "YouTube ChatBot":
        initialize_session_state_youtube()
    elif option == "Video ChatBot":
        initialize_session_state_video()
    elif option == "Website ChatBot":
        initialize_session_state_website()

# def conversation_chat_document(query, chain, history):
#     result = chain({"question": query, "chat_history": history})
#     history.append((query, result["answer"]))
#     return result["answer"]

# def conversation_chat_youtube(query, chain, history):
#     result = chain({"question": query, "chat_history": history})
#     history.append((query, result["answer"]))
#     return result["answer"]

# def conversation_chat_video(query, chain, history):
#     result = chain({"question": query, "chat_history": history})
#     history.append((query, result["answer"]))
#     return result["answer"]


def conversation_chat_document(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def conversation_chat_youtube(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def conversation_chat_video(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def conversation_chat_website(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history_document(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form_document', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input_document')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat_document(user_input, chain, st.session_state['history_document'])

            st.session_state['past_document'].append(user_input)
            st.session_state['generated_document'].append(output)

    if st.session_state['generated_document']:
        with reply_container:
            for i in range(len(st.session_state['generated_document'])):
                message(st.session_state["past_document"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated_document"][i], key=str(i), avatar_style="fun-emoji")


def display_chat_history_youtube(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form_youtube', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your YouTube Video", key='input_youtube')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat_youtube(user_input, chain, st.session_state['history_youtube'])

            st.session_state['past_youtube'].append(user_input)
            st.session_state['generated_youtube'].append(output)

    if st.session_state['generated_youtube']:
        with reply_container:
            for i in range(len(st.session_state['generated_youtube'])):
                message(st.session_state["past_youtube"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated_youtube"][i], key=str(i), avatar_style="fun-emoji")

                
def display_chat_history_video(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form_video', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your video", key='input_video')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat_video(user_input, chain, st.session_state['history_video'])

            st.session_state['past_video'].append(user_input)
            st.session_state['generated_video'].append(output)

    if st.session_state['generated_video']:
        with reply_container:
            for i in range(len(st.session_state['generated_video'])):
                message(st.session_state["past_video"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated_video"][i], key=str(i), avatar_style="fun-emoji")


def display_chat_history_website(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form_website', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about website", key='input_website')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat_website(user_input, chain, st.session_state['history_website'])

            st.session_state['past_website'].append(user_input)
            st.session_state['generated_website'].append(output)

    if st.session_state['generated_website']:
        with reply_container:
            for i in range(len(st.session_state['generated_website'])):
                message(st.session_state["past_website"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated_website"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain_document(vector_store):
    load_dotenv()
    # Create llm
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    # llm = Replicate(
    #         streaming = True,
    #         model = "meta/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         # model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         callbacks=[StreamingStdOutCallbackHandler()],
    #         model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1}
    #     )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(  llm=llm, 
                                                    chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)
    return chain

def create_conversational_chain_youtube(vector_store):
    load_dotenv()
    # Create llm
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    # llm = Replicate(
    #         streaming = True,
    #         model = "meta/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         # model = "meta/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         callbacks=[StreamingStdOutCallbackHandler()],
    #         model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1}
    #     )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(  llm=llm, 
                                                    chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)
    return chain


def create_conversational_chain_video(vector_store):
    load_dotenv()
    # Create llm

    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})

    # llm = Replicate(
    #         streaming = True,
    #         model = "meta/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         callbacks=[StreamingStdOutCallbackHandler()],
    #         model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1}
    #     )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(  llm=llm, 
                                                    chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)
    return chain


def create_conversational_chain_website(vector_store):
    load_dotenv()
     # Create llm
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q5_K_M.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    # Create llm
    # llm = Replicate(
    #         streaming = True,
    #         model = "meta/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
    #         callbacks=[StreamingStdOutCallbackHandler()],
    #         model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1}
    #     )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(  llm=llm, 
                                                    chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)
    return chain

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple Things", page_icon=":books:")
    st.title("InfoInteract: Chat with Multiple Things :books:")

    st.sidebar.title("Choose Your Action")
    input_choice = st.sidebar.radio("Select what you want to do:", ("Document ChatBot", "YouTube Video ChatBot", "Video ChatBot", "Website ChatBot"))

    # Clear and initialize session state when radio button selection changes
    if "prev_radio_selection" not in st.session_state:
        st.session_state["prev_radio_selection"] = input_choice

    if input_choice != st.session_state["prev_radio_selection"]:
        clear_and_initialize_session_state(input_choice)
        st.session_state["prev_radio_selection"] = input_choice

    # Document Chatbot
    if input_choice == "Document ChatBot":
        initialize_session_state_document()

        st.sidebar.title("Document Processing")
        uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "txt"])

        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path, extract_images=True)
                elif file_extension == ".docx" or file_extension == ".doc":
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
            text_chunks = text_splitter.split_documents(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # Create the chain object
            chain = create_conversational_chain_document(vector_store)

            
            display_chat_history_document(chain)

    # Youtube Chatbot
    elif input_choice == "YouTube Video ChatBot":
        initialize_session_state_youtube()

        st.sidebar.title("YouTube Video ChatBot")
        video_link = st.sidebar.text_input("Enter YouTube Video Link")
        
        if video_link:
            summary = get_yt_summary_test(video_link)
            # st.write(summary)
            yt_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50, length_function=len)
            yt_text_chunks = yt_text_splitter.split_text(summary)
            chunks = yt_text_splitter.create_documents(yt_text_chunks)

            # st.write(chunks)

            # Create embeddings
            yt_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            yt_vector_store = FAISS.from_documents(chunks, embedding=yt_embeddings)

            # Create the chain object
            yt_chain = create_conversational_chain_youtube(yt_vector_store)

            
            display_chat_history_youtube(yt_chain)
    
    # Video Chatbot
    elif input_choice == "Video ChatBot":
        initialize_session_state_video()

        st.sidebar.title("Video ChatBot")
        uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4"])
        
        if uploaded_video:
            summary = get_video_text(uploaded_video)

            # st.write(summary)

            video_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50, length_function=len)
            video_text_chunks = video_text_splitter.split_text(summary)
            chunks = video_text_splitter.create_documents(video_text_chunks)

            # st.write(chunks)

            # Create embeddings
            video_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            video_vector_store = FAISS.from_documents(chunks, embedding=video_embeddings)

            # Create the chain object
            video_chain = create_conversational_chain_video(video_vector_store)

            
            display_chat_history_video(video_chain)

    # Website Chatbot
    elif input_choice == "Website ChatBot":
        initialize_session_state_website()

        st.sidebar.title("Website ChatBot")
        website_link = st.sidebar.text_input("Enter Website Link")
        
        if website_link:
            loader = WebBaseLoader(website_link)
            document = loader.load()
            # st.write(document)
            website_text_splitter = CharacterTextSplitter()
            website_text_chunks = website_text_splitter.split_documents(document)

            # st.write(website_text_chunks)

            # Create embeddings
            website_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            website_vector_store = FAISS.from_documents(website_text_chunks, embedding=website_embeddings)

            # Create the chain object
            website_chain = create_conversational_chain_website(website_vector_store)

            
            display_chat_history_website(website_chain)
    
if __name__ == "__main__":
    main()
