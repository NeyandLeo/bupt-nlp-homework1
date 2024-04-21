from collections import Counter
import numpy as np
import random
from scipy.spatial.distance import cosine
from multiprocessing import Pool
import pickle
import json

def load_data(training_path, max_length=5000000):
    with open(training_path, 'r', encoding='utf-8') as file:
        text = file.read().lower().split()
    return text[:max_length]

def create_vocab(text, min_freq=0):
    vocab = Counter(text)
    # 过滤低频词，并创建词汇表
    vocab = {word: idx+1 for idx, (word, freq) in enumerate(vocab.items()) if freq >= min_freq}  # 从1开始编号
    vocab['<pad>'] = 0  # 添加填充符，ID为0
    with open('vocab/vocab_sgns.json', 'w') as fw:
        json.dump(vocab, fw)
    return vocab

def get_negative_sampling_distribution(vocab, text):
    # 根据文本生成负采样分布，只包括词汇表里的词
    total_count = sum([vocab[word] for word in text if word in vocab])
    word_freqs = {word: count / total_count for word, count in Counter(text).items() if word in vocab}
    word_probs = {word: freq ** 0.75 for word, freq in word_freqs.items()}
    word_probs['<pad>'] = 0
    # 确保概率分布的总和为1
    total_prob = sum(word_probs.values())
    normalized_probs = {word: prob / total_prob for word, prob in word_probs.items()}
    with open('vocab/normalized_probs_sgns.json', 'w') as fw:
        json.dump(normalized_probs, fw)
    return normalized_probs

def get_context_positions(i, window_size, text_length):
    return list(range(max(0, i - window_size), min(text_length, i + window_size + 1)))

def process_word(args):
    i, text, vocab, window_size = args  # 移除不再需要的参数
    word = text[i]
    if word in vocab:
        context_positions = get_context_positions(i, window_size, len(text))
        context_words = [text[j] for j in context_positions if j != i and text[j] in vocab]
        context_indices = [vocab[w] for w in context_words]
        while len(context_indices) < 2 * window_size:
            context_indices.append(vocab['<pad>'])
        
        word_index = vocab[word]
        return (word_index, context_indices)
    return None

def generate_training_data_parallel(text, vocab, window_size):
    args = [(i, text, vocab, window_size) for i in range(len(text))]  # 移除不再需要的参数
    with Pool() as pool:
        training_data = pool.map(process_word, args)

    # Filter out None values and save the training data to a file
    training_data = [data for data in training_data if data is not None]
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    return training_data

if __name__ == "__main__":
    text = load_data("data/lmtraining.txt")
    print("load_text_done")
    vocab = create_vocab(text)
    print("vocab_done")
    word_probs = get_negative_sampling_distribution(vocab,text)
    print("word_probs_done")
    training_data = generate_training_data_parallel(text, vocab, window_size=2)