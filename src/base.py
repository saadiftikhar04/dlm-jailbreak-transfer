"""
This file saves the abstract class for all LLMs, DLMs, and remote-APIs, to prevent circular imports.
"""
class LLM:
    def __init__(self):
        pass
    
    def infer(self):
        pass


class DLM:
    def __init__(self):
        pass
    
    def infer(self):
        pass


class API:
    def __init__(self):
        pass
    
    def call(self):
        pass
