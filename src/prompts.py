from langchain_core.prompts import ChatPromptTemplate

from src.langfuse import langfuse

PROMPTS = {
    "base_template":
        """Answer the question to your best knowledge when looking at the following context:
        {context}

        Question: {question}
        """
}


def get_prompt(key: str, from_langfuse: bool) -> ChatPromptTemplate:
    if from_langfuse:
        langfuse_prompt = langfuse.get_prompt(key)
        return ChatPromptTemplate.from_template(langfuse_prompt.get_langchain_prompt())

    return ChatPromptTemplate.from_template(PROMPTS[key])
