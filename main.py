from dotenv import load_dotenv                          # Dotenv for loading environment variables.
import os                                               # OS module for interacting with the operating system.
import asyncio                                          # for callback only

from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm


def load_video(video_id):
    loader = YoutubeLoader(video_id=video_id)
    document = loader.load()
    return document

def split_text(document, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        # separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    texts = text_splitter.split_documents(document)
    # num_tokens = llm.get_num_tokens(texts)
    # print(f'Has {num_tokens} tokens')
    print(texts)
    return texts

def init_llm(temperature):
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    llm=OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
        model_name="text-davinci-003"
    )
    return llm

def init_llm(temperature):
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    llm=OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
        model_name="text-davinci-003"
    )
    return llm

def summary_document(llm, texts, showTokenUsage):
    with get_openai_callback() as cb:
        summary_chain = load_summarize_chain(
            llm, 
            chain_type="stuff", 
            verbose=True
        )

        tx = summary_chain.invoke(texts[:2])
        print(tx)
        
        # num_tokens = llm.get_num_tokens(texts[5])
        # print(f'Has {num_tokens} tokens')
        # exit(0)

        # if len(texts) >= 3000:  # This model's maximum context length is 4097 tokens
        #     tx = chain.run(texts[:2])
        # else:
        #     tx = chain.run(texts)

        if showTokenUsage:    
            # print token usage
            print('---' * 20)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
        
def main():
    if not load_dotenv():
        print("Could not load .env file or it is empty. Please check if it exists and is readable.")
        exit(1)     # The call exit(0) indicates successful execution of a program whereas exit(1) indicates some issue/error occurred while executing a program. 

    print(">>> Youtube Loading...\n")
    document = load_video("Onzd5QxKaGQ")
    
    print(">>> text splitter...\n")
    texts = split_text(document, 2000, 20)
    
    print(">>> llm initiate...\n")
    llm=init_llm(0)
    
    print(">>> llm load_summarize_chain...\n")
    summary_document(llm, texts, showTokenUsage=True)

if __name__ == '__main__':
    main()