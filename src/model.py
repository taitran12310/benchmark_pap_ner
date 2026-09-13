"""Transformer encoder + optional CRF head for token-level NER."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from crf import CRF


class TransformerCrfForNer(nn.Module):
    """PLM encoder + linear projection + optional CRF (PAP_NER hybrid setup)."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        use_crf: bool = True,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        try:
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
        except Exception:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden, num_labels)

        self.crf: Optional[CRF]
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None

    def _prepare_crf_tensors(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """Slice past leading -100 (usually [CLS]) so CRF mask[:,0] is valid.

        After B→I subword fill, active content is contiguous until [SEP]/pad.
        """
        start = 0
        if labels is not None and labels.size(1) > 0 and bool((labels[:, 0] == -100).all()):
            start = 1
        elif labels is None and emissions.size(1) > 1:
            # Standard PLM: position 0 is [CLS]
            start = 1

        em = emissions[:, start:]
        attn = attention_mask[:, start:]
        lab = labels[:, start:] if labels is not None else None

        if lab is not None:
            mask = attn.bool() & (lab != -100)
            # Guarantee first timestep on for CRF API (pad-only rows → force True)
            if mask.size(1) > 0:
                empty = ~mask.any(dim=1)
                mask = mask.clone()
                mask[empty, 0] = True
            tags = lab.clone()
            tags[lab == -100] = 0
            return em, mask, tags, start

        mask = attn.bool()
        if mask.size(1) > 0:
            empty = ~mask.any(dim=1)
            mask = mask.clone()
            mask[empty, 0] = True
        return em, mask, None, start

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        decode: bool = True,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None and self.config.model_type not in {
            "roberta",
            "xlm-roberta",
            "camembert",
            "phobert",
        }:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        result: Dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is None:
            if decode:
                result["predictions"] = self.decode(emissions, attention_mask, labels=None)
            return result

        if self.use_crf and self.crf is not None:
            em, mask, tags, _start = self._prepare_crf_tensors(
                emissions, attention_mask, labels
            )
            assert tags is not None
            # CRF.forward returns log-likelihood; training minimizes NLL (>= 0)
            log_likelihood = self.crf(em, tags, mask=mask, reduction="mean")
            result["loss"] = -log_likelihood
            if decode:
                result["predictions"] = self.decode(
                    emissions, attention_mask, labels=labels
                )
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
            result["loss"] = loss
            if decode:
                result["predictions"] = emissions.argmax(dim=-1)
        return result

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        if not (self.use_crf and self.crf is not None):
            return emissions.argmax(dim=-1)

        em, mask, _tags, start = self._prepare_crf_tensors(
            emissions, attention_mask, labels
        )
        paths = self.crf.decode(em, mask=mask)
        decoded = emissions.new_full((batch_size, seq_len), 0, dtype=torch.long)
        for i, path in enumerate(paths):
            # Scatter path back after leading special tokens
            length = min(len(path), seq_len - start)
            if length <= 0:
                continue
            decoded[i, start : start + length] = torch.tensor(
                path[:length], dtype=torch.long, device=emissions.device
            )
        return decoded

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
