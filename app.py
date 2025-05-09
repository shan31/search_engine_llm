import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from duckduckgo_search import DuckDuckGoSearchException
import time


import os
from dotenv import load_dotenv
load_dotenv()

## Arxiv and Wikipedia API Wrappers
api_wrapper_arxiv =  ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=250,
    description="Search Arxiv for a query and return the first result.",
)

arxiv = ArxivQueryRun(
    api_wrapper=api_wrapper_arxiv,
    description="Search Arxiv for a query and return the first result.",
    name="Arxiv",
    return_only_outputs=True,
)

api_wrapper_wikipedia = WikipediaAPIWrapper(
    top_k_results=1,doc_content_chars_max=250
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper_wikipedia,
    description="Search Wikipedia for a query and return the first result.",
    name="Wikipedia",
    return_only_outputs=True,
)

search = DuckDuckGoSearchRun(
    name="Search",
    description="Search DuckDuckGo for a query and return the first result.",
)

# Wrap the underlying _run method to handle rate limits

original_search_run = search._run



def throttled_search_run(query):

    delay = 1

    for attempt in range(3):

        try:

            time.sleep(delay)

            return original_search_run(query)

        except DuckDuckGoSearchException as e:

            st.sidebar.info(f"[Search retry {attempt+1}] Rate limited, retrying in {delay}s.")

            delay *= 2

    return "Search service temporarily unavailable due to rate limits."



search._run = throttled_search_run

st.title("Langchain Agent with Groq LLM")
# """
# ## This is a simple Langchain agent that uses the Groq LLM to answer questions.
# ## In this example, we will use the Arxiv and Wikipedia APIs to search for information.
# ## The agent will use the DuckDuckGo search engine to find information on the web.
# ## In this example we are using StreamlitCallbackHandler to stream the response from the LLM.
# """

st.title("Q&A Chatbot with OpenAI and LangChain")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API KEY", type="password") 


if  "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is Machine Learning?"):    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192",streaming=True)

    tools = [arxiv, wiki, search]

    search_agent = initialize_agent(tools,llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True, verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

        