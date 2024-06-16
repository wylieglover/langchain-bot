import sys

from io import StringIO
from typing import Dict, Optional
from setup import *
from langchain.agents.tools import Tool

class AutoTool:
    def get_tool(self, **kwargs):
        doc_lines = self.__class__.__doc__.split('\n')
        class_name = doc_lines[0]                    
        class_desc = "\n".join(doc_lines[1:]).strip() 
        
        return Tool(
            name        = kwargs.get('name',        class_name),
            description = kwargs.get('description', class_desc),
            func        = kwargs.get('func',        self.run),
        )
    
    def run(self, command: str) -> str:
        return command
    
class AskForInputTool(AutoTool):
    """Ask-For-Input Tool
    
    This tool asks the user for input, which you can use to gather more information. 
    Use only when necessary, since their time is important and you want to give them a great experience! For example:
    Action-Input: What is your name?
    """
    def __init__(self, fn = input):
        self.fn = fn
    
    def run(self, command: str) -> str:
        response = self.fn(command)
        return response
