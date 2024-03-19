import os
from enum import Enum

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

load_dotenv()


class LLMModel(Enum):
    GPT_3_AZURE = "gpt-3-azure"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


def get_llm_model(model: LLMModel = LLMModel.GPT_3_AZURE):
    if model == LLMModel.GPT_3_5_TURBO:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model_name="gpt-3.5-turbo")

    if model == LLMModel.GPT_3_AZURE:
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        return AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )

    raise ValueError(f"Model {model} not supported.")