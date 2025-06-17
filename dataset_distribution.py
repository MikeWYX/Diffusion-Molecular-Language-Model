import datasets
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd

def analyze_molecule_lengths():
    tokenizer = AutoTokenizer.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
    dataset = datasets.load_dataset('/NAS/luoyc/wuyux/data/zinc250k', split='train')
    lengths = []
    
    print("Analyzing sequence lengths...")
    for item in tqdm(dataset):
        tokens = tokenizer.encode(item['selfies'], add_special_tokens=False)
        lengths.append(len(tokens))
    lengths = np.array(lengths)

    stats = {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'percentile_95': np.percentile(lengths, 95),
        'percentile_99': np.percentile(lengths, 99)
    }

    print("\nSequence Length Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
        
    plt.figure(figsize=(12, 6))
    sns.histplot(data=lengths, bins=50)
    plt.title('Distribution of Molecule Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig('length_distribution.png')
    print("\nLength distribution plot saved as 'length_distribution.png'")
    
    with open('length_statistics.txt', 'w') as f:
        f.write("Sequence Length Statistics:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
        f.write("\nPercentile Information:\n")
        percentiles = range(0, 101, 5)
        for p in percentiles:
            value = np.percentile(lengths, p)
            f.write(f"{p}th percentile: {value:.2f}\n")
    
    length_counts = pd.Series(lengths).value_counts().sort_index()
    length_dist_df = pd.DataFrame({
        'length': length_counts.index,
        'count': length_counts.values,
        'percentage': length_counts.values / len(lengths) * 100
    })
    
    length_dist_df.to_csv('length_distribution.csv', index=False)
    print("\nDetailed length distribution saved to 'length_distribution.csv'")
    
    return stats, lengths

if __name__ == "__main__":
    stats, lengths = analyze_molecule_lengths()
