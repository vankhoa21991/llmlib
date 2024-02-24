from llm import LLM
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LLMcpp(LLM):
    def load(self, model_id: str, hf_auth: str, device: str="cuda:0", **kwargs):

        if stream := kwargs.get("stream"):
            callbacks = [StreamingStdOutCallbackHandler()]
        else:
            callbacks = []

        if config := kwargs.get("config"):
            config = config
        else:
            config = {'max_new_tokens': 256, 'temperature': 0.1, 'repetition_penalty': 1.1}

        self.model = CTransformers(model=model_id,
                            model_type=kwargs.get("model_type") or "llama",
                            config=config,
                            callbacks=callbacks,
                            )

    def getHFpipeline(self):
        raise NotImplementedError


    def run(self, prompt: PromptTemplate, **kwargs):
            chain = LLMChain(prompt=prompt, llm=self.model)
            response = chain.run(**kwargs)
            return response

if __name__ == "__main__":
    llm = LLMcpp(model_name="phi2")
    llm.load(model_id="/media/vankhoa/code/public/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf",
                      hf_auth=None, device="cuda:0", stream=True
                      )

    template = """
    	Context: {context}
    	Question: {question}

    		Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    question = "Why Elon Musk is rich?"
    context = "give me a short explanation"

    response =llm.run(question= question, context= context, prompt=prompt)
    print(response)