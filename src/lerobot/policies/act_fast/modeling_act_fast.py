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
"""ACT-style policy that predicts FAST action tokens."""

from __future__ import annotations

import logging
import math
from collections import deque
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from scipy.fft import idct
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import ACTEncoder, ACTSinusoidalPositionEmbedding2d
from lerobot.policies.act_fast.configuration_act_fast import ACTFastConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)
from lerobot.utils.import_utils import _transformers_available

if _transformers_available:
    from transformers import AutoProcessor, AutoModel, AutoConfig
else:
    AutoProcessor = None
    AutoModel = None
    AutoConfig = None


class TransformersBackbone(nn.Module):
    def __init__(self, model_name, frozen=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.num_features = self.config.hidden_size

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        last_hidden_state = outputs.last_hidden_state
        # DINOv3 has 1 CLS token and num_register_tokens (default 4)
        num_special_tokens = 1 + getattr(self.config, "num_register_tokens", 0)
        patch_tokens = last_hidden_state[:, num_special_tokens:]

        B, L, C = patch_tokens.shape
        H = int(math.sqrt(L))
        W = H

        if H * W != L:
             raise ValueError(f"Feature map size mismatch: {L} is not a perfect square (H={H}, W={W}).")

        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        return {"feature_map": feature_map}


class ACTFastPolicy(PreTrainedPolicy):
    """
    ACT-style policy that predicts discrete FAST action tokens and detokenizes to action chunks.
    """

    config_class = ACTFastConfig
    name = "act_fast"

    def __init__(self, config: ACTFastConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTFast(config)
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        self.action_tokenizer = None
        if AutoProcessor is not None and self.config.action_tokenizer_name:
            try:
                self.action_tokenizer = AutoProcessor.from_pretrained(
                    self.config.action_tokenizer_name, trust_remote_code=True
                )
            except Exception as exc:
                logging.warning(f"Failed to load FAST tokenizer for detokenization: {exc}")

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return self.config.fast_vocab_size + 1

    @property
    def eos_token_id(self) -> int:
        return self.config.fast_vocab_size + 2

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        action_tokens = self.sample_action_tokens(
            batch,
            max_steps=self.config.max_decoding_steps,
            temperature=self.config.temperature,
        )

        action_dim = self.config.output_features[ACTION].shape[0]
        actions = self.detokenize_actions(action_tokens, action_horizon=self.config.chunk_size, action_dim=action_dim)
        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        tokens = batch.get(ACTION_TOKENS)
        masks = batch.get(ACTION_TOKEN_MASK)
        if tokens is None or masks is None:
            raise ValueError(
                f"ACTFast requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} to be present in the batch"
            )

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            masks = masks.unsqueeze(0)

        encoder_out, encoder_pos_embed = self.model.encode(batch)

        input_tokens = tokens[:, :-1]
        input_masks = masks[:, :-1]
        target_tokens = tokens[:, 1:]
        target_masks = masks[:, 1:]

        logits = self.model.decode_tokens(encoder_out, encoder_pos_embed, input_tokens, input_masks)

        vocab_size = logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_flat = loss_fct(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
        loss_per_token = loss_flat.reshape(target_tokens.shape)
        masked_loss = loss_per_token * target_masks.float()

        denom = target_masks.sum(dim=1).clamp(min=1)
        per_sample_loss = masked_loss.sum(dim=1) / denom
        loss = per_sample_loss.mean()

        loss_dict = {"ce_loss": loss.item()}
        if reduction == "none":
            return per_sample_loss, loss_dict
        return loss, loss_dict

    @torch.no_grad()
    def sample_action_tokens(
        self, batch: dict[str, Tensor], max_steps: int | None = None, temperature: float | None = None
    ) -> Tensor:
        if max_steps is None:
            max_steps = self.config.max_decoding_steps or self.config.max_action_tokens
        if temperature is None:
            temperature = self.config.temperature

        max_steps = min(max_steps, self.config.max_action_tokens)

        encoder_out, encoder_pos_embed = self.model.encode(batch)
        batch_size = encoder_out.shape[1]
        device = encoder_out.device

        tokens = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        masks = torch.ones_like(tokens, dtype=torch.bool)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_steps - 1):
            logits = self.model.decode_tokens(encoder_out, encoder_pos_embed, tokens, masks)
            next_logits = logits[:, -1, :]

            if temperature is None or temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            masks = torch.cat([masks, torch.ones_like(next_token[:, None], dtype=torch.bool)], dim=1)
            finished |= next_token == self.eos_token_id
            if finished.all():
                break

        return tokens

    def decode_actions_with_fast(
        self, token_ids: list[list[int]], time_horizon: int, action_dim: int, relaxed_decoding: bool = True
    ) -> np.ndarray:
        if self.action_tokenizer is None:
            raise ValueError("Action tokenizer not initialized. Provide action_tokenizer_name in config.")

        decoded_actions = []
        for token_seq in token_ids:
            try:
                decoded_tokens = self.action_tokenizer.bpe_tokenizer.decode(token_seq)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.action_tokenizer.min_token

                if relaxed_decoding:
                    expected_seq_len = time_horizon * action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (time_horizon, action_dim), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, "
                    f"expected ({time_horizon}, {action_dim})"
                )
            except Exception as exc:
                logging.warning(f"Error decoding tokens: {exc}")
                logging.warning(f"Tokens: {token_seq}")
                decoded_dct_coeff = np.zeros((time_horizon, action_dim))

            decoded_actions.append(
                idct(decoded_dct_coeff / self.action_tokenizer.scale, axis=0, norm="ortho")
            )

        return np.stack(decoded_actions)

    def detokenize_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        if tokens is None:
            raise ValueError("Tokens cannot be None")

        single_sample = tokens.dim() == 1
        if single_sample:
            tokens = tokens.unsqueeze(0)

        cleaned_tokens = []
        for seq in tokens:
            seq_list = [t for t in seq.tolist() if t != self.pad_token_id]
            seq_list = [t for t in seq_list if t not in (self.bos_token_id, self.eos_token_id)]
            seq_list = [t - 1 for t in seq_list if t > 0]
            cleaned_tokens.append(seq_list)

        actions = self.decode_actions_with_fast(
            cleaned_tokens, time_horizon=action_horizon, action_dim=action_dim
        )
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=tokens.device)

        if single_sample:
            actions_tensor = actions_tensor.squeeze(0)
        return actions_tensor


class ACTFast(nn.Module):
    """ACT-style observation encoder + causal token decoder for FAST tokens."""

    def __init__(self, config: ACTFastConfig):
        super().__init__()
        self.config = config

        if self.config.image_features:
            if "resnet" in self.config.vision_backbone:
                backbone_model = getattr(torchvision.models, self.config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, self.config.replace_final_stride_with_dilation],
                    weights=self.config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                for param in backbone_model.parameters():
                    param.requires_grad = False
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
                backbone_num_features = backbone_model.fc.in_features
            else:
                self.backbone = TransformersBackbone(self.config.vision_backbone)
                backbone_num_features = self.backbone.num_features

        self.encoder = ACTEncoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_num_features, config.dim_model, kernel_size=1
            )

        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.feedforward_activation,
            batch_first=False,
        )
        self.token_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        self.token_embedding = nn.Embedding(config.vocab_size, config.dim_model)
        self.token_pos_embedding = nn.Embedding(config.max_action_tokens, config.dim_model)
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.token_decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
            device = batch[OBS_IMAGES][0].device
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]
            device = batch[OBS_ENV_STATE].device

        latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=device)

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_out, encoder_in_pos_embed

    def decode_tokens(
        self,
        encoder_out: Tensor,
        encoder_pos_embed: Tensor,
        token_ids: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)

        token_emb = self.token_embedding(token_ids) + self.token_pos_embedding(positions)[None, :, :]
        token_emb = token_emb.transpose(0, 1)  # (T, B, D)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=token_ids.device), diagonal=1).bool()
        key_padding_mask = ~token_mask

        memory = encoder_out + encoder_pos_embed
        decoder_out = self.token_decoder(
            token_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )

        decoder_out = decoder_out.transpose(0, 1)
        logits = self.lm_head(decoder_out)
        return logits
