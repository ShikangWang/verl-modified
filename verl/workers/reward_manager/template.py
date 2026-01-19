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
from collections import Counter
from nltk.util import ngrams
import spacy
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0
from langdetect import detect_langs
import re

def pos_tag(text: str, nlp) -> str:
    """返回空格分隔的POS序列"""
    return " ".join([tok.pos_ for tok in nlp(text)])

def is_english_langdetect(text: str, min_confidence: float = 0.6) -> bool:
    """
    使用langdetect检测是否为英文
    min_confidence: 最小置信度
    """
    if not text.strip():
        return False
    
    # 清理文本
    clean_text = re.sub(r'\s+', ' ', text.strip())
    if len(clean_text) < 10:  # 太短的文本不可靠
        return False
    
    # 检测语言
    try :
        languages = detect_langs(clean_text)
    except LangDetectException:
        print(f"Language detection failed. {text[:30]}...")
        return False
    
    # 检查是否有英语
    for lang in languages:
        if lang.lang == 'en' and lang.prob >= min_confidence:
            return True
    
    return False

@register("template")
class TemplateRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the TemplateRewardManager instance.

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
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        self.ngram_counter = Counter()
        self.min_freq = 4  # 最小频次阈值
        self.top_k = 100  # 只考虑前top_k高频模板

    def __call__(self, data: DataProto, return_dict: bool = False, beta=0.6) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                responses = data.non_tensor_batch["responses"]
                responses_POS = [pos_tag(resp, self.nlp) for resp in responses]
                original_rewards = data.batch["rm_scores"].sum(dim=-1).unsqueeze(-1) # B, 1
                data.batch['src_rm_scores'] = data.batch['rm_scores'].clone()

                is_english = torch.tensor([is_english_langdetect(resp) for resp in responses], device=original_rewards.device).bool()

                filtered = {tpl: c for tpl, c in self.ngram_counter.items() if c >= self.min_freq}
                top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
                top = set([" ".join(tpl) for tpl, _ in top])
                # import pdb
                # if len(top) > 0 :
                #     pdb.set_trace()
                template_cnt = [sum([t in resp for t in top]) for resp in responses_POS]
                template_cnt = torch.tensor(template_cnt, device=original_rewards.device).float()
                print(f"template_cnt: {template_cnt}")
                #template_cnt越大，diversity_score越低
                available_cnt = template_cnt[is_english]
                if len(available_cnt) == 0:
                    mean_cnt = 0.0
                    std_cnt = 0.0
                else:
                    mean_cnt = available_cnt.float().mean()
                    std_cnt = available_cnt.float().std()

                # 确保约68%的样本在合理范围
                alpha = 1.0 / (mean_cnt + std_cnt + 1e-8)
                diversity_score = torch.exp(-alpha * template_cnt)
                
                diversity_score = diversity_score * 2 - 1
                diversity_score[~is_english] = 0.0  # 非英文样本不计算多样性分数

                data.batch["diversity_scores"] = diversity_score
                print(f"diversity_score: {diversity_score.squeeze(-1)}")

                sign = torch.sign(original_rewards)
                # 处理零值
                sign[sign == 0] = 1.0  # 零reward当作正reward处理

                diversity_factor = 1.0 + sign * beta * diversity_score
                final_reward = original_rewards * diversity_factor
                data.batch["rm_scores"] = final_reward

                for i in range(len(responses_POS)):
                    if is_english[i]:
                        self.ngram_counter.update(ngrams(responses_POS[i].split(), 6))

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
