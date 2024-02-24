from llm import LLM
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline

class LLMfull(LLM):
    def load(self, model_id: str, hf_auth: str, device: str="cuda:0"):
        config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.init_device = device

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True,
            use_auth_token=hf_auth,
        ).to(device)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

    def getHFpipeline(self, model_id: str, hf_auth: str, device: str, **kwargs):
        self.load(model_id=model_id, hf_auth=hf_auth, device=device)
        if stream := kwargs.get("streamer"):
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        else:
            streamer = None
        generate_text = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=256,  # max number of tokens to generate in the output
            repetition_penalty=1.1,  # without this output begins repeating
            torch_dtype=torch.float16,
            device='cuda:0',
            streamer=streamer
        )

        self.hfpipe = HuggingFacePipeline(pipeline=generate_text)
        return self.hfpipe

    def run(self, prompt: PromptTemplate, **kwargs):
            chain = LLMChain(prompt=prompt, llm=self.hfpipe)
            response = chain.run(**kwargs)
            return response

if __name__ == "__main__":
    llm = LLMfull(model_name="phi2")
    llm.getHFpipeline(model_id="/media/vankhoa/code/public/phi-2", hf_auth=None, device="cuda:0", streamer=True)

    template = """
    	Context: {context}
    	Question: {question}

    		Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    question = "Why Elon Musk is rich?"
    context = "give me a short explanation"

    response =llm.run(question= question, context= context, prompt=prompt)
    print(response)