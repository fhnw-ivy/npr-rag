from langfuse import Langfuse

langfuse = Langfuse()

langfuse.auth_check()

class Evaluator:
    def __init__(self):
        self.langfuse = langfuse

    def evaluate(self, code: str, language: str, input: str = ""):
        return self.langfuse.evaluate(code, language, input)