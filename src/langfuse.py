import os
import string
from random import random

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()
langfuse = Langfuse()

class DatasetItem:
    def __init__(self, input_str, output_str):
        self.input_str = input_str
        self.output_str = output_str

    def __repr__(self):
        return f"IOData(input={self.input!r}, output={self.output!r})"

def generate_session_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=30))

def get_langfuse_handler():
    # TODO: move callback handler to context specific call to track on a per chat basis
    # this could be done by creating a class that can be passed to the chain handling context
    # https://langfuse.com/docs/tracing/sessions
    langfuse_trace = langfuse.trace(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
        # session_id=generate_session_id(), # TODO: implement session id tracking
        user_id=None  # TODO: implement user id tracking
    )

    handler = langfuse_trace.get_langchain_handler()

    assert langfuse.auth_check()
    assert handler.auth_check()

    return handler

def create_new_dataset(name: str):
    langfuse.create_dataset(name=name)

def add_item_to_dataset(dataset_name: str, item: DatasetItem):
    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input={
            "text": item.input_str
        },
        expected_output={
            "text": item.output_str
        }
    )

# TODO: implemnent possibility to score traces
# https://langfuse.com/docs/sdk/python#scores
