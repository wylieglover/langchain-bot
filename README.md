# LangChain-Bot
A chat bot capable of using tools and remembering chat history with conversation buffers

# Local Development Setup
Before starting, these components must be installed (preferably in an enviornment):
- cuda v12.1 
- python
- pip

Contents of requirements.txt:
- wheel
- transformers
- langchain
- langchain-community
- pydantic
- torch=2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
- accelerate
- optimum
- auto-gptq

Start by cloning this repository into a local folder/directory:
```sh
git clone https://github.com/wylieglover/langchain-bot.git
```

Navigate into the repository's folder and run the command below to download the necessary components:
```sh
pip install -r requirements.txt
```

Then finally run the ```agent.py``` file:
```sh
python3 agent.py
```
