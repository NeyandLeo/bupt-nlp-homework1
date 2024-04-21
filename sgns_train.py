import torch
from torch import nn, optim
from collections import Counter
from torch.utils.data import DataLoader
import numpy as np
import random
from scipy.spatial.distance import cosine
import pickle
import os
import json
from SGNSmodel.sgns_model import SkipGramModel
from SGNSmodel.sgns_dataset import TrainingDataset

def load_vocab(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:  
            # 使用json.load()方法解析JSON数据  
            vocab = json.load(file) 
    return vocab

def load_probs(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:  
            # 使用json.load()方法解析JSON数据  
            probs = json.load(file) 
    return probs

def get_negative_sampling_distribution(vocab, text):
    # 根据文本生成负采样分布，只包括词汇表里的词
    total_count = sum([vocab[word] for word in text if word in vocab])
    word_freqs = {word: count / total_count for word, count in Counter(text).items() if word in vocab}
    word_probs = {word: freq ** 0.75 for word, freq in word_freqs.items()}
    word_probs['<pad>'] = 0
    # 确保概率分布的总和为1
    total_prob = sum(word_probs.values())
    normalized_probs = {word: prob / total_prob for word, prob in word_probs.items()}
    return normalized_probs

def train_model(model, dataset, batch_size, epochs, learning_rate, vocab, word_probs, num_negatives):
    print("start_training")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    vocab_list = list(vocab.keys())
    probs = [word_probs[word] for word in vocab_list]
    
    step_count = 0

    for epoch in range(epochs):
        total_loss = 0
        try:
            for center, context in dataloader:
                center, context = center.to(device), context.to(device)
                negatives = [np.random.choice(len(vocab_list), size=num_negatives, replace=True, p=probs) for _ in range(center.size(0))]
                negatives = [[vocab[vocab_list[idx]] for idx in batch] for batch in negatives]
                negatives = torch.tensor(negatives).to(device)

                optimizer.zero_grad()
                positive_score, negative_score = model(center, context, negatives)
                positive_loss = -torch.nn.functional.logsigmoid(positive_score).mean()
                negative_loss = -torch.nn.functional.logsigmoid(-negative_score).mean()
                loss = positive_loss + negative_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                step_count += 1
                if step_count % 1000 == 0:
                    print(f"Step {step_count}, Current Loss: {loss.item()}")

            print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")
        except RuntimeError as e:
            print(f"Runtime error in epoch {epoch+1}: {e}")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")# 例如，训练结束后保存模型

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def evaluate_model(model, vocab, test_path):
    model.eval()  # Set the model to evaluation mode
    # Calculate the sum of center and context word embeddings
    embeddings = model.centerword_embeddings.weight.detach().cpu().numpy() + model.contextword_embeddings.weight.detach().cpu().numpy()
    
    # Open both the input and output files within the same 'with' statement
    with open(test_path, 'r', encoding='utf-8') as file, open('sgns_test.txt', 'w', encoding='utf-8') as outfile:
        # Read all lines from the input file
        lines = file.readlines()

        # Process each line in the input file
        for line in lines:
            words = line.strip().split()
            word1, word2, human_score = words[1], words[2], words[3]
            if word1 in vocab and word2 in vocab:
                vec1 = embeddings[vocab[word1]]
                vec2 = embeddings[vocab[word2]]
                sim_sgns = cosine_similarity(vec1, vec2)  # Calculate similarity
            else:
                sim_sgns = 0  # Default similarity if words are not in vocab
            # Write the output to outfile within the 'with' block
            outfile.write(f"{line.strip()}\t{sim_sgns}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_vocab = "vocab/vocab_sgns.json"
    vocab = load_vocab(path_vocab)
    print("vocab_loaded")
    path_probs = "vocab/normalized_probs_sgns.json"
    word_probs = load_probs(path_probs)
    print("probs_loaded")
    # Usage
    dataset = TrainingDataset('data/training_data.pkl')
    embedding_dim = 100
    model = SkipGramModel(len(vocab), embedding_dim)
    # Train the model
    batch_size = 512
    epochs = 3
    learning_rate = 0.01
    num_negatives=5
    model = train_model(model, dataset, batch_size, epochs, learning_rate,vocab,word_probs,num_negatives)

    path = "./SGNSmodel.pth"
    save_model(model, path)


    # 评估训练后的sgns模型，可以注释掉
    #results = evaluate_model(model, vocab, "wordsim353_agreed.txt")
