# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from sentence_transformers import SentenceTransformer


@register("multi")
class MultiRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, data: DataProto, return_dict: bool = False, beta=0.3) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                responses = data.non_tensor_batch["responses"]
                original_rewards = data.batch["rm_scores"].sum(dim=-1).unsqueeze(-1) # B, 1
                data.batch['src_rm_scores'] = data.batch['rm_scores'].clone()

                # 计算多样性分数
                with torch.no_grad():
                    sentence_embeddings = self.sentence_model.encode(responses, batch_size=8, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
                    # cosine_scores = torch.cosine_similarity(sentence_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=-1)
                    cosine_scores = torch.mm(sentence_embeddings, sentence_embeddings.T)  # 计算余弦相似度矩阵
                    mask = torch.eye(cosine_scores.size(0), device=cosine_scores.device).bool()
                    cosine_scores.masked_fill_(mask, 0.0)  # 将对角线元素设为0，避免与自身比较
                    cosine_scores = cosine_scores.sum(dim=-1) / (cosine_scores.size(0) - 1)  # 计算平均相似度

                
                diversity_score = 1 - cosine_scores  # 越不相似，多样性分数越高
                diversity_score = diversity_score.unsqueeze(-1)
                data.batch["diversity_scores"] = diversity_score
                print(f"diversity_scores: {diversity_score.squeeze(-1)[:10]}")

                data.batch["rm_scores"] = original_rewards

                #方法二
                # final_reward = torch.sigmoid(original_rewards) + diversity_score * beta
                # data.batch["rm_scores"] = final_reward

                # print(data.batch["rm_scores"].shape, data.batch["src_rm_scores"].shape)
                # print(f"reward_tensor: {data.batch['rm_scores']}, reward_extra_info: {reward_extra_info}")
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
