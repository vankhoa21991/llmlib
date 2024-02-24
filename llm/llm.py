import transformers
import torch
from langchain.llms import HuggingFacePipeline

# create an abstract class for all types of Large Language Models
class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self, model_id: str, hf_auth: str, device: str):
        raise NotImplementedError

    def run(self, question: str, context: str, past: str):
        raise NotImplementedError


