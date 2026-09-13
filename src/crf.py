"""Linear-chain CRF (batch_first) — no external pytorch-crf dependency."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """
    Conditional Random Field for sequence labeling.

    API similar to pytorch-crf:
      - forward(emissions, tags, mask) -> log-likelihood (sum/mean)
      - decode(emissions, mask) -> List[List[int]]
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        super().__init__()
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return log-likelihood of `tags` given `emissions`."""
        emissions, tags, mask = self._validate(emissions, tags=tags, mask=mask)
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        if reduction == "token_mean":
            return llh.sum() / mask.float().sum()
        raise ValueError(f"invalid reduction: {reduction}")

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        emissions, mask = self._validate(emissions, mask=mask)[:2]
        return self._viterbi_decode(emissions, mask)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if emissions.dim() != 3:
            raise ValueError(f"emissions must have shape (B,T,C), got {tuple(emissions.shape)}")

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)

        batch_size, seq_length, num_tags = emissions.shape
        if num_tags != self.num_tags:
            raise ValueError(f"expected {self.num_tags} tags, got {num_tags}")

        if mask is None:
            mask = emissions.new_ones(batch_size, seq_length, dtype=torch.bool)
        else:
            if mask.dim() != 2:
                raise ValueError(f"mask must be (B,T), got {tuple(mask.shape)}")
            if not self.batch_first:
                mask = mask.transpose(0, 1)
            mask = mask.bool()

        if tags is not None:
            if tags.dim() != 2:
                raise ValueError(f"tags must be (B,T), got {tuple(tags.shape)}")
            if not self.batch_first:
                tags = tags.transpose(0, 1)

        # First timestep must be valid for every sequence
        if not mask[:, 0].all():
            raise ValueError("mask of the first timestep must all be on")

        if tags is None:
            return emissions, mask
        return emissions, tags, mask

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_length):
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            score += (trans + emit) * mask[:, i].float()

        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)  # (B, C, 1)
            broadcast_emit = emissions[:, i].unsqueeze(1)  # (B, 1, C)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        batch_size, seq_length, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emit = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        seq_ends = mask.long().sum(dim=1) - 1
        best_tags_list: List[List[int]] = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list
