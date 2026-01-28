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
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act_fast.configuration_act_fast import ACTFastConfig
from lerobot.policies.act_fast.modeling_act_fast import ACTFastPolicy
from lerobot.processor import FastActionTokenizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import ACTION, ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_ENV_STATE


class DummyFastTokenizer:
    def __init__(self, decoded_len: int = 8):
        self.min_token = 0
        self.scale = 1.0
        self._decoded_len = decoded_len

        class DummyBPE:
            def __init__(self, decoded_len):
                self.decoded_len = decoded_len

            def decode(self, _token_seq):
                return "".join([chr(0) for _ in range(self.decoded_len)])

        self.bpe_tokenizer = DummyBPE(decoded_len)

    def __call__(self, _action):
        return [1, 2, 3]


def test_fast_action_tokenizer_step_outputs_tokens():
    step = FastActionTokenizerProcessorStep(
        action_tokenizer_input_object=DummyFastTokenizer(),
        max_action_tokens=6,
        fast_vocab_size=10,
    )
    transition = {TransitionKey.ACTION: torch.zeros(4, 2)}
    processed = step(transition)
    comp = processed[TransitionKey.COMPLEMENTARY_DATA]

    assert ACTION_TOKENS in comp
    assert ACTION_TOKEN_MASK in comp
    assert comp[ACTION_TOKENS].shape[-1] == 6
    assert comp[ACTION_TOKEN_MASK].shape[-1] == 6


def test_act_fast_detokenize_shape():
    config = ACTFastConfig(
        input_features={
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        chunk_size=4,
        n_action_steps=4,
        fast_vocab_size=10,
        max_action_tokens=6,
        action_tokenizer_name=None,
    )
    policy = ACTFastPolicy(config)
    policy.action_tokenizer = DummyFastTokenizer(decoded_len=8)

    tokens = torch.tensor(
        [
            policy.bos_token_id,
            2,
            3,
            policy.eos_token_id,
            policy.pad_token_id,
            policy.pad_token_id,
        ]
    )

    actions = policy.detokenize_actions(tokens, action_horizon=4, action_dim=2)
    assert actions.shape == (4, 2)
