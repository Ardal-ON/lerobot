#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS
from lerobot.utils.import_utils import _transformers_available

from .core import EnvTransition, TransitionKey
from .pipeline import ActionProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


@dataclass
@ProcessorStepRegistry.register(name="fast_action_tokenizer_processor")
class FastActionTokenizerProcessorStep(ActionProcessorStep):
    """
    Tokenize action chunks using a FAST action tokenizer (raw FAST token IDs).
    """

    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_vocab_size: int = 1024
    # Internal tokenizer instance (not part of the config)
    action_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use FastActionTokenizerProcessorStep."
            )

        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object
        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return self.fast_vocab_size + 1

    @property
    def eos_token_id(self) -> int:
        return self.fast_vocab_size + 2

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        tokens, mask = self._tokenize_action(action)

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary_data is None:
            complementary_data = {}
        complementary_data[ACTION_TOKEN_MASK] = mask
        complementary_data[ACTION_TOKENS] = tokens
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action is None:
            raise ValueError("Action cannot be None")

        device = action.device if isinstance(action, torch.Tensor) else None

        single_sample = action.dim() == 2
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]
        tokens_list = []
        masks_list = []

        for i in range(batch_size):
            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)

            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:
                tokens = tokens.to(device=action.device)

            if tokens.dim() > 1:
                tokens = tokens.flatten()

            # Shift tokens to reserve 0 for PAD.
            tokens = tokens + 1

            tokens = torch.cat(
                [
                    torch.tensor([self.bos_token_id], device=action.device),
                    tokens,
                    torch.tensor([self.eos_token_id], device=action.device),
                ]
            )

            if len(tokens) > self.max_action_tokens:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
                    "Consider increasing `max_action_tokens` if this happens frequently."
                )
                tokens = tokens[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(
                            self.max_action_tokens - len(tokens), dtype=torch.bool, device=action.device
                        ),
                    ]
                )
                tokens = torch.nn.functional.pad(
                    tokens, (0, self.max_action_tokens - len(tokens)), value=self.pad_token_id
                )

            tokens_list.append(tokens)
            masks_list.append(mask)

        tokens_batch = torch.stack(tokens_list, dim=0)
        masks_batch = torch.stack(masks_list, dim=0)

        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)

        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)

        return tokens_batch, masks_batch

    def action(self, action: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._tokenize_action(action)
        return tokens

    def get_config(self) -> dict[str, Any]:
        config = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
            "fast_vocab_size": self.fast_vocab_size,
        }
        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name
        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """No change to feature definitions."""
        return features
