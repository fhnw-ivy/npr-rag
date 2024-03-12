from dotenv import load_dotenv
from langfuse.callback import CallbackHandler

load_dotenv()
from langfuse import Langfuse

langfuse = Langfuse()


class LangfuseHandler:
    def __init__(self):
        # TODO: pass relevant trace information
        # like mode (production/eval) etc.
        self.handler = CallbackHandler()
        assert self.handler.auth_check()

    def get_callback_handler(self):
        return self.handler

    def _get_trace_id(self):
        return self.handler.get_trace_id()

    def score(self, name, value, comment=None):
        _ = langfuse.score(
            trace_id=self._get_trace_id(),
            name=name,
            value=value,
            comment=comment
        )
