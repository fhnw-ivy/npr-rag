from enum import Enum

from dotenv import load_dotenv
from langfuse import Langfuse

from src.config import SYSTEM_CONFIG

load_dotenv()

langfuse = Langfuse(
    release=SYSTEM_CONFIG["release"]
)


class TraceTag(Enum):
    production = "production"
    dev = "dev"
    eval = "eval"


class TraceManager:
    def __init__(self,
                 version: str,
                 tags: [TraceTag] = None,
                 metadata: dict = None):

        self.trace = langfuse.trace(
            version=version,
            name="qa",
            tags=[tag.value for tag in tags] if tags else None,
            metadata=metadata
        )

    def get_callback_handler(self):
        return self.trace.get_langchain_handler()

    def add_query(self, query: str):
        self.trace.update(input=query)

    def add_output(self, output: str):
        self.trace.update(output=output)

    def score(self, name, value, comment=None):
        self.trace.score(
            name=name,
            value=value,
            comment=comment
        )
