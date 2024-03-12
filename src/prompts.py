from langchain_core.prompts import ChatPromptTemplate

from src.langfuse import langfuse

PROMPTS = {
    "base_template":
        ("""Answer the question to your best knowledge when looking at the following context:
        {context}

        Question: {question}
        """, 1)
}


class Prompt:
    def __init__(self,
                 key: str,
                 template: ChatPromptTemplate,
                 version: int = None):

        self.key = key
        self.version = version
        self.template = template

    @staticmethod
    def get(key: str, version: int = None, from_langfuse: bool = True):
        if from_langfuse:
            if version is not None:
                langfuse_prompt = langfuse.get_prompt(key, version=version)
            else:
                langfuse_prompt = langfuse.get_prompt(key)

            template = ChatPromptTemplate.from_template(langfuse_prompt.get_langchain_prompt())
            return Prompt(key, template, langfuse_prompt.version)

        prompt = PROMPTS[key]
        return Prompt(key, ChatPromptTemplate.from_template(prompt[0]), prompt[1])

