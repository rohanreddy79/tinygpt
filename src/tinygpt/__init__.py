"""tinygpt: a decoder-only transformer trained from scratch, readable end to end.

Public surface:
- :mod:`tinygpt.tokenizer` -- byte-level BPE trained from scratch.
- :mod:`tinygpt.model` -- ``GPTConfig`` and the ``GPT`` module with KV-cache generation.
- :mod:`tinygpt.train` -- data loader, AdamW groups, warmup+cosine schedule, checkpoints.
- :mod:`tinygpt.cli` -- the ``tinygpt`` command line (prepare / train / sample / plot).
"""

from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import Tokenizer

__all__ = ["GPT", "GPTConfig", "Tokenizer"]
__version__ = "0.1.0"
