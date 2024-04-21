# 自然语言处理作业-词向量的构建
本项目使用Skip-Gram Negative Sampling（SGNS）方法以及SVD方法来构建词向量。

## 项目结构

- `/data`：包含用于数据生成和预处理的数据文件和脚本。
  - `datageneration.py`：用于生成和预处理数据的脚本。
  - `datageneration_parallel.py`：与datageneration.py功能完全相同，区别在于使用了多线程来加速数据集的构造。
  - `lmtraining.txt`：用于语言模型训练的文本数据。
  - `training_data.pkl`：序列化的Python对象，包含训练数据。
  - `wordsim353_agreed.txt`：词相似度数据集，用于模型的评估。

- `/SGNSmodel`：包含定义SGNS模型和数据集处理的脚本。
  - `sgns_dataset.py`：定义SGNS模型的训练所需的数据集类。
  - `sgns_model.py`：包含SGNS模型架构的实现。

- `/vocab`：包含模型的词汇表和概率分布文件。
  - `normalized_probs_sgns.json`：SGNS的标准化概率分布文件。
  - `vocab_sgns.json`：SGNS模型的词表文件。
  - `vocab_svd.json`：SVD模型的词表文件。

- `2021213640.txt`：模型在评估语料上的测试结果

- `evaluate.py`：评估脚本，用于对两个模型在评估语料的评估。

- `README.md`：项目文档。

- `sgns_train.py`：用于训练SGNS模型的脚本。

- `SGNSmodel.pth`：保存的SGNS模型权重。

- `svd_train.py`：用于训练SVD模型的脚本。

- `word_embeddings.npy`：包含词嵌入的Numpy数组文件。

## 附录——SGNS模型训练日志及SVD模型训练输出结果
SGNS模型训练日志：
Step 1000, Current Loss: 0.9220963716506958
Step 2000, Current Loss: 0.3517785668373108
Step 3000, Current Loss: 0.2586981952190399
Step 4000, Current Loss: 0.17055517435073853
Step 5000, Current Loss: 0.10608211159706116
Step 6000, Current Loss: 0.12856270372867584
Step 7000, Current Loss: 0.08878012001514435
Step 8000, Current Loss: 0.08954407274723053
Step 9000, Current Loss: 0.08056534826755524
Epoch 1, Average Loss: 0.47034401010520516
Step 10000, Current Loss: 0.013001250103116035
Step 11000, Current Loss: 0.018802154809236526
Step 12000, Current Loss: 0.004910028539597988
Step 13000, Current Loss: 0.016379347071051598
Step 14000, Current Loss: 0.022405153140425682
Step 15000, Current Loss: 0.006077175494283438
Step 16000, Current Loss: 0.02491256408393383
Step 17000, Current Loss: 0.0273663979023695
Step 18000, Current Loss: 0.017455609515309334
Step 19000, Current Loss: 0.02228156290948391
Epoch 2, Average Loss: 0.01765987244743487
Step 20000, Current Loss: 0.0008678666781634092
Step 21000, Current Loss: 0.0011565589811652899
Step 22000, Current Loss: 0.0008880642708390951
Step 23000, Current Loss: 0.00624270411208272
Step 24000, Current Loss: 1.1203900612599682e-05
Step 25000, Current Loss: 0.0011919193202629685
Step 26000, Current Loss: 0.003854426322504878
Step 27000, Current Loss: 0.002253257902339101
Step 28000, Current Loss: 0.0003881170123349875
Step 29000, Current Loss: 0.004955930169671774
Epoch 3, Average Loss: 0.003301385168382063

SVD模型训练输出结果：
Total number of nonzero singular values: 21681
Number of selected singular values: 213
Sum of selected singular values: 528361.31
Sum of all singular values: 754769.00
Ratio of selected to total singular values: 0.70