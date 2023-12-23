from dotenv import load_dotenv                          # Dotenv for loading environment variables.
import os                                               # OS module for interacting with the operating system.
import asyncio                                          # for callback only

from langchain.llms.openai import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

from tqdm import tqdm

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

print(">>> Youtube Loading...\n")
loader = YoutubeLoader(video_id="Onzd5QxKaGQ")
result = loader.load()

print(">>> text splitter...\n")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
texts = text_splitter.split_documents(result)
# num_tokens = llm.get_num_tokens(texts)
# print(f'Has {num_tokens} tokens')
print(texts)

print(">>> llm initiate...\n")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm=OpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

print(">>> llm load_summarize_chain...\n")
with get_openai_callback() as cb:
    chain = load_summarize_chain(
        llm, 
        chain_type="stuff", 
        verbose=True
    )

    tx = chain.run(texts[:2])
    print(tx)
    
    # num_tokens = llm.get_num_tokens(texts[5])
    # print(f'Has {num_tokens} tokens')
    # exit(0)

    # if len(texts) >= 3000:  # This model's maximum context length is 4097 tokens
    #     tx = chain.run(texts[:2])
    # else:
    #     tx = chain.run(texts)

 
    # print token usage   
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    
# # Import chat-related modules from LangChain.
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )

# def init():
#     load_dotenv()  # Load environment variables from a .env file.

#     # Load the OpenAI API key from an environment variable.
#     if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPEN_API_KEY") == "":
#         print(">> OPENAI_API_KEY is not set")
#         exit(1)
#     else: 
#         print(">> OPENAI_API_KEY is set")
    
#     # Define Streamlit page configuration.
#     st.set_page_config(
#         page_title="Your own ChatGPT",  # Title of the page.
#         page_icon="⌨️"                   # Icon of the page.
#     )

# def main():
#     init()  # Call the initialization function.
    
#     # Create an instance of ChatOpenAI from LangChain
#     chat = ChatOpenAI(temperature=0.5)  # Temperature parameter determines the creativity of the conversation.
    
#     # Initialize the Streamlit session state to keep track of the messages.
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             SystemMessage(content="You are a helpful assistant.")  # Start the conversation with a system message.
#         ]
    
#     # Set up the header of the page.
#     st.header("Your own ChatGPT.")
    
#     # Create a text input field in the sidebar for user input.
#     with st.sidebar:
#         user_input = st.text_input("Your message: ", key="user_input")
#         show_message = st.radio(
#             "Set a message visibility 👉",
#             key="visibility",
#             options=["visible", "hidden"],
#             index=1,
#         )
    
#     # Process the user's message.
#     if user_input:
#         st.session_state.messages.append(HumanMessage(content=user_input))  # Add the user's message to the session state.
#         with st.spinner("Thinking..."):                                     # Show a spinner while waiting for a response.
#             response = chat(st.session_state.messages)                      # Get a response from the LangChain chatbot.
#         st.session_state.messages.append(AIMessage(content=response.content))  # Add the AI's response to the session state.

#     # Iterate through the stored messages and display them on the screen.
#     messages = st.session_state.get('messages', [])
    
#     for i, msg in enumerate(messages[1:]):                              # Display all messages except the first system message.
#         if i % 2 == 0:
#             stchat_message(msg.content, is_user=True, key=str(i) + '_user')    # Display user messages.
#         else:
#             stchat_message(msg.content, is_user=False, key=str(i) + '_ai')     # Display AI messages.

#     if show_message == "visible":
#         st.write(messages)

# if __name__ == '__main__':
#     main()
    