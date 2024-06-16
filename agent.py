from tools import *
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import TransformChain, SequentialChain, LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.agents import BaseSingleActionAgent
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms import BaseLLM

from typing import List, Tuple, Any, Union, Optional
from pydantic.v1 import root_validator, Field
from abc import abstractmethod

llama_full_prompt = PromptTemplate.from_template(
    template="<s>[INST]<<SYS>>{sys_msg}<</SYS>>\n\nContext:\n{history}\n\nHuman: {input}\nHuman Emotion based off message: {user_emotion} \nHuman Toxicity based off message: {user_toxicity}\n[/INST] {primer}",
)

llama_prompt = llama_full_prompt.partial(
    sys_msg = (
        "You are a helpful, respectful and honest AI assistant."
        "\nAlways answer as helpfully as possible, while being safe."
        "\nPlease be brief and efficient unless asked to elaborate, and follow the conversation flow."
        "\nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        "\nEnsure that your responses are socially unbiased and positive in nature."
        "\nIf a question does not make sense or is not factually coherent, explain why instead of answering something incorrect."
        "\nIf you don't know the answer to a question, please don't share false information."
        "\nIf the user asks for a format to output, please follow it as closely as possible."
    ),
    primer = "",
    history = "",
    user_toxicity = "",
    user_emotion = ""
)

img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
emo_pipe = pipeline('sentiment-analysis', 'SamLowe/roberta-base-go_emotions')
zsc_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tox_pipe = pipeline("text-classification", model="nicholasKluge/ToxicityModel")

def generate(prompt, max_length=1024, pipe=llama_pipe, **kwargs):
    def_kwargs = dict(return_full_text=False, return_dict=False)
    response = llama_pipe(prompt.strip(), max_length=max_length, **kwargs, **def_kwargs)
    return response[0]['generated_text']

class MyAgentBase(BaseSingleActionAgent):
    @root_validator
    def validate_input(cls, values: Any) -> Any:
        return values
    
    @abstractmethod
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any): 
        pass

    def action(self, tool, tool_input, finish=False) -> Union[AgentAction, AgentFinish]:
        if finish: return AgentFinish({"output": tool_input},           log = f"\nFinal Answer: {tool_input}\n")
        else:      return AgentAction(tool=tool, tool_input=tool_input, log = f"\nAgent: {tool_input.strip()}\n")
        # else:    return AgentAction(tool=tool, tool_input=tool_input, log = f"\nTool: {tool}\nInput: {tool_input}\n") ## Actually Correct
    
    async def aplan(self, intermediate_steps, **kwargs):
        return await self.plan(intermediate_steps, **kwargs)
    
    @property
    def input_keys(self):
        return ["input"]
    
class MyAgent(MyAgentBase):
    general_prompt : PromptTemplate
    llm            : BaseLLM

    general_chain  : Optional[LLMChain]
    max_messages   : int                   = Field(5, gt=1)

    temperature    : float                 = Field(0.6, gt=0, le=1)
    max_new_tokens : int                   = Field(128, ge=1, le=2048)
    eos_token_id   : Union[int, List[int]] = Field(2, ge=0)
    gen_kw_keys = ['temperature', 'max_new_tokens', 'eos_token_id']
    gen_kw = {}

    user_toxicity  : float = 0.5
    user_emotion   : str = "Unknown"

    @root_validator
    def validate_input(cls, values: Any) -> Any:
        '''Think of this like the BaseModel's __init__ method'''
        if not values.get('general_chain'):
            llm = values.get('llm')
            prompt = values.get("general_prompt")
            prompt.input_variables = ['input', 'history']

            values['general_chain'] = ConversationChain(llm=llm, prompt=prompt, verbose=True, memory=ConversationBufferMemory(memory_key="history", ai_prefix="AI")) 
        values['gen_kw'] = {k:v for k,v in values.items() if k in values.get('gen_kw_keys')}
        return values


    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any):
        tool, response = "Ask-For-Input Tool", "Hello World! How can I help you?"
        if len(intermediate_steps) == 0:
            return self.action(tool, response)

        ## History of past agent queries/observations
        queries      = [step[0].tool_input for step in intermediate_steps]
        observations = [step[1]            for step in intermediate_steps]
        last_obs     = observations[-1]    # Most recent observation (i.e. user input)

        self.user_toxicity = tox_pipe(last_obs)[0]['score']
        self.user_emotion = emo_pipe(last_obs)[0]['label']

        self.general_chain.prompt.partial_variables['user_toxicity'] = f"{self.user_toxicity:.4f}"
        self.general_chain.prompt.partial_variables['user_emotion'] = self.user_emotion

        if len(observations) >= self.max_messages:
            response = "Thanks so much for the chat, and hope to see ya later! Goodbye!"
            return self.action(tool, response, finish=True)

        image_links = [part for part in last_obs.split('`') if part.endswith(('.png', '.jpg', '.jpeg'))]
        if image_links:
            image_link = image_links[0]
            image_caption = img_pipe(image_link)[0]['generated_text']
            response = f"I see an image. Here's a description: {image_caption}"
            return self.action(tool, response)

        if "```" in last_obs:
            response = generate(f"""
                                <s>[INST] <<SYS>>
                                You are a system that can only respond with valid python code.
                                You should not generate any discussion output unless it is contained within a comment block.
                                Do not include any superfluous content, as the responses will go into a REPL editor with no modification.
                                <</SYS>>

                                Please provide a basic hello-world application!
                                [/INST]
                                ```python
                                ## Prints out hello world; defaults to a display in the standard output
                                print("Hello World!")
                                ```
                                </s><s>[INST]
                                {last_obs}
                                [/INST]
                                ```python
                                """, do_sample=True, eos_token_id=[2, 421, 4954, 7521, 16159, 28956, 29952]).replace('```', '')
            return self.action(tool, response.strip())

        with SetParams(llm, **self.gen_kw):
            response = self.general_chain.invoke(last_obs)

        return self.action(tool, response)

    def reset(self):
        self.user_toxicity = 0
        self.user_emotion = "Unknown"
        if getattr(self.general_chain, 'memory', None) is not None:
            self.general_chain.memory.clear()


converser = input 

agent_kw = dict(
    llm = llm,
    general_prompt = llama_prompt,
    max_new_tokens = 128,
    eos_token_id = [2]   
)

agent_ex = AgentExecutor.from_agent_and_tools(
    agent = MyAgent(**agent_kw),
    tools=[AskForInputTool(converser).get_tool()], 
    verbose=True
)

try: agent_ex.invoke("")
except KeyboardInterrupt:
    print("Cleaning up resources...")
    # Clear CUDA cache
    torch.cuda.empty_cache()
    print("Cleanup completed.")
    exit(0)