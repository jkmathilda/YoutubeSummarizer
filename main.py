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
    