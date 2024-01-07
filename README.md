# YoutubeSummarizer

YoutubeSummarizer is one application of LLM, which offers a unique service in the world of video content. It specializes in providing concise video summaries of YouTube videos, ensuring users can swiftly access key insights without the need to watch the entire content. 

YoutubeSumarizer divides video content manageable segments, facilitating efficient analysis of even lengthy videos. Then, retrieval mechanisms generate informative summaries, enhancing accessibility and saving user's valuable time. 

# Getting Started

To get started with this project, you'll need to clone the repository and set up a virtual environment. This will allow you to install the required dependencies without affecting your system-wide Python installation.

### Cloning the Repository

    git clone https://github.com/jkmathilda/gpt-YoutubeSummarizer.git

### Setting up a Virtual Environment

    cd ./gpt-YoutubeSummarizer

    pyenv versions

    pyenv local 3.11.6

    echo '.env'  >> .gitignore
    echo '.venv' >> .gitignore

    ls -la

    python -m venv .venv        # create a new virtual environment

    source .venv/bin/activate   # Activate the virtual environment

    python -V                   # Check a python version

### Install the required dependencies

    pip list

    pip install -r requirements.txt

    pip freeze | tee requirements.txt.detail

### Configure the Application

To configure the application, there are a few properties that can be set the environment

    echo 'OPENAI_API_KEY="sk-...."' >> .env

### Running the Application

    python main.py

### Deactivate the virtual environment

    deactivate

### Reference

Youtube : Summarize a YouTube Video + Count OpenAI token Usage
[https://www.youtube.com/watch?v=rA3RtNV797I]

