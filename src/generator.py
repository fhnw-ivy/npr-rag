import openai
from templates import base_template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vectorstore import VectorStore

class Generator:
    def __init__(self, openai_api_key: str, template: str = base_template, vectorstore: VectorStore=None):
        self.openai_api_key = openai_api_key
        self.template = template
        self.vectorstore = vectorstore
        self.retriever = vectorstore.get_retriever()
        openai.api_key = self.openai_api_key

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