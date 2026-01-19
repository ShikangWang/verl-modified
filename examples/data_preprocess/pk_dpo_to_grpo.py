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
"""
转换具有chosen/rejected格式的数据到GRPO格式
"""

import argparse
import os
import json
from typing import Dict, List, Any, Optional
import re
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs



def convert_dpo_to_grpo(input_data: Dict[str, Any], data_source: str, split: str, idx: int) -> Dict[str, Any]:
    """
    将DPO格式数据转换为GRPO格式
    
    Args:
        input_data: 输入的DPO格式数据
        data_source: 数据源名称
        split: 数据集划分（train/test等）
        idx: 样本索引
    
    Returns:
        转换后的GRPO格式数据
    """
    conversations = input_data.get("conversations", [])
    chosen = input_data.get("chosen", {}).get("value", "")
    rejected = input_data.get("rejected", {}).get("value", "")
    
    # 构建提示
    prompt = []
    for conv in conversations:
        if conv["from"] == "system":
            prompt.append({
                "role": "system",
                "content": conv["value"]
            })
        elif conv["from"] == "human":
            prompt.append({
                "role": "user",
                "content": conv["value"]
            })
        elif conv["from"] == "gpt":
            prompt.append({
                "role": "assistant",
                "content": conv["value"]
            })
    
    # 获取最后一个human消息作为问题
    question_raw = ""
    for conv in reversed(conversations):
        if conv["from"] == "human":
            question_raw = conv["value"]
            break
    
    if len(prompt) == 0:
        return None

    # 构建GRPO格式数据
    grpo_data = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "RolePlay",  # 可以根据数据特性调整
        "reward_model": {
            "style": "rule",
            "ground_truth": chosen  # 使用chosen作为标准答案
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "chosen": chosen,
            "rejected": rejected,
            "question": question_raw
        }
    }
    
    return grpo_data


def process_file(input_path: str, output_path: str, data_source: str, split: str = "train") -> None:
    """
    处理输入文件并转换为GRPO格式，保存为parquet格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        data_source: 数据源名称
        split: 数据集划分
    """
    import pandas as pd
    
    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    
    grpo_data_list = []
    print(f"len of dataset{len(data_list)}")
    for idx, item in tqdm(enumerate(data_list), total=len(data_list), desc="Processing"):
        grpo_item = convert_dpo_to_grpo(item, data_source, split, idx)
        if grpo_item is not None:
            grpo_data_list.append(grpo_item)
    
    # 转换为DataFrame并保存为parquet
    df = pd.DataFrame(grpo_data_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved {len(grpo_data_list)} records to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='/archive/workspace/eval/verl/data/chaiting_allpk_new1125_100r_truncted0-1900_family_score_filter.json', help="输入的DPO格式数据文件")
    parser.add_argument("--output_file", default=None, help="输出的GRPO格式数据文件")
    parser.add_argument("--data_source", default="dpo_data", help="数据源名称")
    parser.add_argument("--split", default="train", help="数据集划分")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS目录")
    
    args = parser.parse_args()

    if args.output_file is None:
        file_name = os.path.basename(args.input_file).split('.')[0]
        args.output_file = os.path.join('/archive/workspace/eval/verl/data', f"grpo_{file_name}.parquet")
    
    process_file(args.input_file, args.output_file, args.data_source, args.split)
    
    # 如果需要保存到HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        output_dir = os.path.dirname(args.output_file)
        copy(src=output_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")
    
    print(f"Processed {args.input_file} to {args.output_file}")