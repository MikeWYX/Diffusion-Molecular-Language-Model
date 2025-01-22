import datasets
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import json

def calculate_generation_distribution(total_samples=10000):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
    
    # 加载数据集
    dataset = datasets.load_dataset('/NAS/luoyc/wuyux/data/zinc250k', split='train')
    
    # 收集长度信息
    lengths = []
    print("Analyzing sequence lengths...")
    for item in tqdm(dataset):
        tokens = tokenizer.encode(item['selfies'], add_special_tokens=False)
        lengths.append(len(tokens))
    
    # 计算长度分布
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    probabilities = counts / len(lengths)
    
    # 计算每个长度需要生成的样本数
    samples_per_length = (probabilities * total_samples).astype(int)
    
    # 确保总数等于要求的样本数
    while samples_per_length.sum() < total_samples:
        # 找到概率最大的长度，增加一个样本
        idx = np.argmax(probabilities)
        samples_per_length[idx] += 1
    
    # 创建长度到样本数的映射
    generation_dist = {int(length): int(samples) for length, samples in zip(unique_lengths, samples_per_length)}
    
    # 保存分布信息
    with open('generation_distribution.json', 'w') as f:
        json.dump(generation_dist, f, indent=4)
    
    print(f"\nGeneration distribution saved to 'generation_distribution.json'")
    print(f"Total samples: {sum(generation_dist.values())}")
    
    return generation_dist

if __name__ == "__main__":
    distribution = calculate_generation_distribution(10000)