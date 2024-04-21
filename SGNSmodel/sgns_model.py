import torch
from torch import nn, optim

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.centerword_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.contextword_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size=vocab_size
        
    def forward(self, center_word, context_words, negative_words):
        center_embed = self.centerword_embeddings(center_word)
        context_embeds = self.contextword_embeddings(context_words)
        negative_embeds = self.contextword_embeddings(negative_words)
        #print(center_embed.size(),context_embeds.size(),negative_embeds.size())
        if center_embed.max() >= self.vocab_size or context_embeds.max() >= self.vocab_size or negative_embeds.max() >= self.vocab_size:
            print("Index out of range error!")
            print("Max center index:", center_word.max())
            print("Max context index:", context.max())
            print("Max negative index:", negatives.max())

        # 计算正样本
        # 计算中心词和正样本上下文词的点积
        positive_score = torch.bmm(center_embed.unsqueeze(1), context_embeds.transpose(1, 2)).squeeze(1)

        # 计算中心词和负样本的点积（使用负样本向量的负值）
        negative_score = torch.bmm(center_embed.unsqueeze(1),negative_embeds.neg().transpose(1, 2)).squeeze(1)

        return positive_score, negative_score