import torch
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 

# 定義模型架構
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        


        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)        
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 16)
        self.linear6 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 2)   

        
    def forward(self, x):
        x = self.linear1(x)
        #print(f'After linear1: {x.shape}')  # 調試輸出
        x = F.relu(x)
        #print(f'After relu1: {x.shape}')  # 調試輸出
        x = self.linear2(x)
        #print(f'After linear2: {x.shape}')  # 調試輸出
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        x = self.linear6(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        #print(f'Final output: {x.shape}')  # 調試輸出
  
        return x

# 定義輸出 CSV 檔案路徑
input_csv_path = r"C:\Users\YuCheng_Ch\Desktop\1731rand_ans_que_merge_tensor.csv" # 替換成你的路徑
model_path = r"C:\Users\YuCheng_Ch\Desktop\datasets_accton\onetime_train\model_epoch_25_acc_83.49_loss_0.4775.pth"
csv_output_path = r'C:\Users\YuCheng_Ch\Desktop\datasets_accton\預測結果\1731rand_ans_que_merge_tensor-predict.csv'
batch_size = 128


df = pd.read_csv(input_csv_path)

# 提取句子和向量

sentences = df['sentence'].tolist()
merged_vectors = df['vector'].apply(eval).tolist()


model = NN()
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.cuda()
model.eval()

# 轉換向量為Tensor
merged_vectors = torch.tensor(merged_vectors, dtype=torch.float32).cuda()

# 使用 DataLoader 進行批量預測
dataset = TensorDataset(merged_vectors)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class_1_probabilities = []

with torch.no_grad():
    for batch in dataloader:
        batch = batch[0].cuda()
        logits = model(batch)
        print(f'Logits: {logits[:5]}')  # 打印前5個 logits
        logits = F.softmax(logits, dim=1)
        print(f'Softmax: {logits[:5]}')  # 打印前5個 softmax 結果
        class_1_probabilities.extend(logits[:, 1].cpu().numpy())



# 儲存結果到CSV
pd.DataFrame({'sentence': sentences, 'class_1_probabilities': class_1_probabilities}).to_csv(csv_output_path, encoding='utf-8-sig', index=False)