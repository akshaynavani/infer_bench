"""Optional HuggingFace transformers backend (install with ``pip install infer-bench[hf]``).

Implements the same prefill / decode_step interface as ``SimulatedLLM`` using a real
causal LM with a manual KV-cache loop. Batched decode pads prompts on the left.
Intended for small CPU-friendly models such as ``sshleifer/tiny-gpt2``.
"""

from __future__ import annotations

from typing import Sequence

from .config import ServerSettings
from .model import Sequence_


class HFBackend:
    def __init__(self, settings: ServerSettings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(settings.hf_model_name)
        self.model.eval()
        self._state: dict[int, dict] = {}

    def new_sequence(self, prompt: str, max_new_tokens: int) -> Sequence_:
        seq = Sequence_(prompt_tokens=[prompt], max_new_tokens=max_new_tokens, seed=0)
        return seq

    def prefill(self, seqs: Sequence[Sequence_]) -> None:
        # Real models carry per-sequence KV caches; we keep the full token ids and
        # recompute from scratch per step for simplicity and batch-shape flexibility.
        for seq in seqs:
            ids = self.tokenizer(seq.prompt_tokens[0], return_tensors="pt").input_ids[0]
            self._state[id(seq)] = {"ids": ids}

    def decode_step(self, seqs: Sequence[Sequence_]) -> list[str]:
        torch = self.torch
        active = [s for s in seqs if not s.done]
        if not active:
            return []
        id_lists = [self._state[id(s)]["ids"].tolist() for s in active]
        batch = self.tokenizer.pad({"input_ids": id_lists}, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**batch).logits[:, -1, :]
        next_ids = torch.argmax(logits, dim=-1)
        out = []
        for seq, nid in zip(active, next_ids.tolist()):
            st = self._state[id(seq)]
            st["ids"] = torch.cat([st["ids"], torch.tensor([nid])])
            tok = self.tokenizer.decode([nid])
            seq.generated.append(tok)
            out.append(tok)
            if seq.done:
                self._state.pop(id(seq), None)
        return out

    def generate_stream(self, prompt: str, max_new_tokens: int):
        seq = self.new_sequence(prompt, max_new_tokens)
        self.prefill([seq])
        while not seq.done:
            (tok,) = self.decode_step([seq])
            yield tok
