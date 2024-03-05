import openai
from templates import base_template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class Generator:
    def __init__(self, openai_api_key: str, template: str = base_template, vectorstore=None):
        self.openai_api_key = openai_api_key
        self.template = template
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever()
        openai.api_key = self.openai_api_key

        print(vectorstore)

    def set_template(self, template: str):
        self.template = template

    def ask(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_template(self.template)

        model = ChatOpenAI()

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        print(f'Answer: {answer}')

        return answer