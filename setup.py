import torch

from transformers import pipeline, logging, GenerationConfig, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from extras_and_licenses.forward_listener import GenerateListener
from accelerate.utils import write_basic_config

write_basic_config(mixed_precision='fp16')
logging.set_verbosity_error()

model_kwargs = {"do_sample": True, "temperature": 0.4, "max_length": 4096}
model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
llama_pipe = pipeline("text-generation", model=model_name, device_map="auto", model_kwargs=model_kwargs)
llm = HuggingFacePipeline(pipeline=llama_pipe)

class SetParams:
    def __init__(self, my_llm, **new_params):
        self.pipeline = my_llm.pipeline
        self._old_params = {**self.pipeline._forward_params}
        self._new_params = new_params

    def __enter__(self):
        self.pipeline._forward_params.update(**self._new_params)

    def __exit__(self ,type, value, traceback):
        for k in self._new_params.keys():
            del self.pipeline._forward_params[k]
        self.pipeline._forward_params.update(self._old_params)
