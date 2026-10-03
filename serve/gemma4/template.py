"""serve/gemma4/template.py - the Gemma 4 chat template, wired into Strata's ChatTemplate.

`serve/frontend.py::ChatTemplate` renders a Jinja template with the same settings transformers uses
and the `tojson`/`raise_exception` globals a chat template expects.  Gemma 4's template needs one
more thing in its context: `bos_token`, which transformers injects from the tokenizer.
`ChatTemplate.render` forwards `**kwargs`, so a thin subclass that supplies `bos_token` (and the
template's other optional knobs) is enough - no change to `frontend.py`.

Which template is used
----------------------
A Gemma 4 GGUF carries its own `tokenizer.chat_template` in metadata, and that is the one the
model was released with (Google revised it after the first upload - the repo's own note says
"re-download for Google's latest chat template").  `from_gguf()` uses the file's template when it
is there and falls back to `chat_template.jinja` when it is not, so the prompt a user sends is the
prompt the model expects.  The bundled `chat_template.jinja` is Google's, byte for byte, not a
re-implementation: a hand-written template for a model this particular (`<|turn>`,
`<|channel>thought`, the tool-call grammar) would be a source of silent wrongness.
"""
from __future__ import annotations

import pathlib

from serve.frontend import ChatTemplate

DEFAULT_TEMPLATE = pathlib.Path(__file__).resolve().parent / "chat_template.jinja"




class Gemma4ChatTemplate(ChatTemplate):
    """`ChatTemplate` + the context Gemma 4's template expects.

    `bos_token` defaults to "" so a tokenizer without a recognised BOS piece still renders (an
    empty BOS is a smaller error than a token the tokenizer cannot encode).  Pass the model's real
    BOS string - e.g. `"<bos>"` - once you have confirmed the tokenizer has that piece."""

    def __init__(self, path: str | pathlib.Path | None = None, *, bos_token: str = "",
                 enable_thinking: bool = False, source: str | None = None):
        super().__init__(path or DEFAULT_TEMPLATE)
        if source is not None:
            # Re-compile inside the environment `ChatTemplate` already built (same sandbox, same
            # `tojson`/`raise_exception`), rather than configuring a second one that could drift.
            self.source = source
            self.template = self.template.environment.from_string(source)
        self._defaults = {"bos_token": bos_token, "enable_thinking": enable_thinking,
                          "preserve_thinking": False}

    @staticmethod
    def from_gguf(gguf_path: str | pathlib.Path, *, bos_token: str = "",
                  enable_thinking: bool = False) -> "Gemma4ChatTemplate":
        """The template the file itself carries, or the bundled one when it carries none."""
        import sys
        root = pathlib.Path(__file__).resolve().parents[2]
        if str(root / "tools") not in sys.path:
            sys.path.insert(0, str(root / "tools"))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(gguf_path)).metadata
        tpl = md.get("tokenizer.chat_template")
        return Gemma4ChatTemplate(bos_token=bos_token, enable_thinking=enable_thinking,
                                  source=tpl if isinstance(tpl, str) and tpl else None)

    def render(self, messages, tools=None, add_generation_prompt: bool = True, **kwargs) -> str:
        # the caller's kwargs win over the defaults (a request that asks for thinking gets it)
        merged = {**self._defaults, **kwargs}
        return super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                              **merged)
