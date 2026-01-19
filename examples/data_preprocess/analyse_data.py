import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_phrases_in_dataset(parquet_file_path):
    """
    分析数据集中特定短语的出现情况
    
    Args:
        parquet_file_path: parquet文件路径
    """
    # 读取数据
    print("正在读取数据...")
    df = pd.read_parquet(parquet_file_path)
    print(f"数据集总样本数: {len(df)}")
    
    # 定义要分析的短语列表
    target_phrases = [
        "can I ask", "can i ask", "may I ask", "may i ask",
        "could I ask", "could i ask",
        "if you don't mind", "if I may", "if i may"
    ]
    
    # 统计结果
    phrase_stats = {}
    total_samples_with_phrases = 0
    out_data_list = []
    
    print("\n正在分析短语出现情况...")
    
    for idx, row in df.iterrows():
        # 获取所有文本内容
        texts_to_check = []
        
        # 检查prompt中的用户消息
        if 'prompt' in row:
            for msg in row['prompt']:
                if msg.get('role') == 'user' and 'content' in msg:
                    texts_to_check.append(msg['content'].lower())
        
        # 检查chosen回答
        if 'extra_info' in row and 'chosen' in row['extra_info']:
            texts_to_check.append(str(row['extra_info']['chosen']).lower())
        
        # 检查rejected回答  
        if 'extra_info' in row and 'rejected' in row['extra_info']:
            texts_to_check.append(str(row['extra_info']['rejected']).lower())
        
        # 检查每个短语
        sample_phrases_found = []
        for text in texts_to_check:
            for phrase in target_phrases:
                if phrase.lower() in text:
                    sample_phrases_found.append(phrase)
        
        # 更新统计
        if sample_phrases_found:
            total_samples_with_phrases += 1
            for phrase in set(sample_phrases_found):  # 去重
                phrase_stats[phrase] = phrase_stats.get(phrase, 0) + 1
        if len(sample_phrases_found) == 0:
            out_data_list.append(row.to_dict())

    
    # 打印详细统计结果
    print("\n" + "="*50)
    print("短语出现统计结果")
    print("="*50)
    
    if phrase_stats:
        # 按出现次数排序
        sorted_phrases = sorted(phrase_stats.items(), key=lambda x: x[1], reverse=True)
        
        for phrase, count in sorted_phrases:
            percentage = (count / len(df)) * 100
            print(f"'{phrase}': {count}次 ({percentage:.2f}%)")
        
        print(f"\n总结:")
        print(f"包含目标短语的样本数: {total_samples_with_phrases}")
        print(f"包含目标短语的样本比例: {(total_samples_with_phrases/len(df))*100:.2f}%")
        print(f"总短语出现次数: {sum(phrase_stats.values())}")
    else:
        print("未找到任何目标短语")

    #save
    print(f"\n保存不包含目标短语的样本到新的parquet文件..., 共{len(out_data_list)}条样本")
    out_df = pd.DataFrame(out_data_list)
    out_df.to_parquet(parquet_file_path.replace('.parquet', '_filtered.parquet'))
    
    return phrase_stats, total_samples_with_phrases

def analyze_phrase_context(parquet_file_path, top_k=5):
    """
    分析包含特定短语的样本上下文
    
    Args:
        parquet_file_path: parquet文件路径
        top_k: 显示前k个样本
    """
    print("\n" + "="*50)
    print("分析包含短语的样本上下文")
    print("="*50)
    
    df = pd.read_parquet(parquet_file_path)
    target_phrases = ["can i ask", "may i ask", "could i ask"]
    
    found_samples = []
    
    for idx, row in df.iterrows():
        # 检查chosen回答
        if 'extra_info' in row and 'chosen' in row['extra_info']:
            chosen_text = str(row['extra_info']['chosen']).lower()
            for phrase in target_phrases:
                if phrase in chosen_text:
                    # 获取上下文
                    context = {
                        'index': idx,
                        'phrase': phrase,
                        'question': row['extra_info'].get('question', '')[:100] + '...' if 'question' in row['extra_info'] else '',
                        'chosen_snippet': chosen_text[max(0, chosen_text.find(phrase)-50):chosen_text.find(phrase)+100] + '...',
                        'full_chosen': chosen_text[:200] + '...' if len(chosen_text) > 200 else chosen_text
                    }
                    found_samples.append(context)
                    break
        
        # 检查rejected回答
        if 'extra_info' in row and 'rejected' in row['extra_info']:
            rejected_text = str(row['extra_info']['rejected']).lower()
            for phrase in target_phrases:
                if phrase in rejected_text:
                    context = {
                        'index': idx,
                        'phrase': phrase,
                        'question': row['extra_info'].get('question', '')[:100] + '...' if 'question' in row['extra_info'] else '',
                        'rejected_snippet': rejected_text[max(0, rejected_text.find(phrase)-50):rejected_text.find(phrase)+100] + '...',
                        'full_rejected': rejected_text[:200] + '...' if len(rejected_text) > 200 else rejected_text
                    }
                    found_samples.append(context)
                    break
    
    # 显示前k个样本
    for i, sample in enumerate(found_samples[:top_k]):
        print(f"\n样本 {i+1}:")
        print(f"  索引: {sample['index']}")
        print(f"  短语: '{sample['phrase']}'")
        print(f"  问题: {sample['question']}")
        if 'chosen_snippet' in sample:
            print(f"  chosen片段: ...{sample['chosen_snippet']}")
        if 'rejected_snippet' in sample:
            print(f"  rejected片段: ...{sample['rejected_snippet']}")
    
    return found_samples

def create_visualization(phrase_stats, output_file=None):
    """
    创建统计可视化
    
    Args:
        phrase_stats: 短语统计字典
        output_file: 输出图片文件路径
    """
    if not phrase_stats:
        print("没有数据可可视化")
        return
    
    # 准备数据
    phrases = list(phrase_stats.keys())
    counts = list(phrase_stats.values())
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 柱状图
    plt.subplot(2, 1, 1)
    bars = plt.bar(phrases, counts, color='skyblue')
    plt.title('目标短语出现次数统计')
    plt.xlabel('短语')
    plt.ylabel('出现次数')
    plt.xticks(rotation=45)
    
    # 在柱子上添加数值
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 饼图
    plt.subplot(2, 1, 2)
    plt.pie(counts, labels=phrases, autopct='%1.1f%%', startangle=90)
    plt.title('短语分布比例')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存到: {output_file}")
    
    plt.show()

def main():
    """主函数"""
    # 替换为您的parquet文件路径
    parquet_file = '/archive/workspace/eval/verl/data/grpo_chaiting_allpk_new1125_100r_truncted0-1900_family_score_filter_100k.parquet'
    
    try:
        # 1. 基本分析
        phrase_stats, total_with_phrases = analyze_phrases_in_dataset(parquet_file)
        
        # 2. 上下文分析
        context_samples = analyze_phrase_context(parquet_file, top_k=3)
        
        # 3. 可视化
        # create_visualization(phrase_stats, 'phrase_analysis.png')
        
        # 4. 生成详细报告
        print("\n" + "="*50)
        print("分析报告总结")
        print("="*50)
        
        df = pd.read_parquet(parquet_file)
        total_samples = len(df)
        
        print(f"数据集: {parquet_file}")
        print(f"总样本数: {total_samples}")
        print(f"包含目标短语的样本数: {total_with_phrases}")
        print(f"包含目标短语的样本比例: {(total_with_phrases/total_samples)*100:.2f}%")
        
        if phrase_stats:
            most_common_phrase = max(phrase_stats.items(), key=lambda x: x[1])
            print(f"最常出现的短语: '{most_common_phrase[0]}' ({most_common_phrase[1]}次)")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()