import torch
import numpy as np
import os
import re
import json
import pickle
from collections import Counter
from scipy.spatial.distance import cosine
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from SGNSmodel.sgns_model import SkipGramModel
from SGNSmodel.sgns_dataset import TrainingDataset

def load_vocab(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:  
            # 使用json.load()方法解析JSON数据  
            vocab = json.load(file) 
    return vocab

def load_embeddings(filename):
    """从文件加载svd嵌入。"""
    return np.load(filename, allow_pickle=True)

def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似性。"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_model(model, vocab, svd_embeddings, vocab_index, test_path):
    """评估模型，计算相似度并写入输出文件。"""
    model.eval()
    embeddings = model.centerword_embeddings.weight.detach().cpu().numpy() + model.contextword_embeddings.weight.detach().cpu().numpy()

    with open(test_path, 'r', encoding='utf-8') as file, open('2021213640.txt', 'w', encoding='utf-8') as outfile:
        for line in file:
            words = line.strip().split()
            word1, word2, human_score = words[1], words[2], words[3]
            sim_svd = cosine_similarity(svd_embeddings[vocab_index[word1]], svd_embeddings[vocab_index[word2]]) if word1 in vocab_index and word2 in vocab_index else 0
            sim_sgns = cosine_similarity(embeddings[vocab[word1]], embeddings[vocab[word2]]) if word1 in vocab and word2 in vocab else 0
            outfile.write(f"{line.strip()}\t{sim_svd}\t{sim_sgns}\n")

def load_sgns_model(model, path):
    """从指定路径加载sgns模型。"""
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# 主脚本执行
if __name__ == "__main__":
    sgns_path ="vocab/vocab_sgns.json"
    svd_path = "vocab/vocab_svd.json"
    model_file = 'SGNSmodel.pth'
    embeddings_file = 'word_embeddings.npy'
    test_file = 'data/wordsim353_agreed.txt'

    vocab_svd = load_vocab(svd_path)
    print("vocab_svd loaded")
    vocab_sgns = load_vocab(sgns_path)
    print("vocab_sgns loaded")
    model = SkipGramModel(vocab_size=len(vocab_sgns), embedding_dim=100)
    sgns_model = load_sgns_model(model, model_file)
    print("sgns model loaded")
    loaded_svd_embeddings = load_embeddings(embeddings_file)
    print("svd embeddings loaded")
    print("start evaluating")
    evaluate_model(sgns_model, vocab_sgns, loaded_svd_embeddings, vocab_svd, test_file)
    print("evaluating done")
