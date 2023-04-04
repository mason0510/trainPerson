from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import gradio as gr
import sys
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain.llms import OpenAIChat
from IPython.display import Markdown, display

from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = 'sk-**'

def construct_index():
    # define LLM
    documents = SimpleDirectoryReader('./graham_essay/data').load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)
    # save
    index.save_to_disk("index.json")
    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Doc AI Chatbot")

index = construct_index()
iface.launch(share=True)
