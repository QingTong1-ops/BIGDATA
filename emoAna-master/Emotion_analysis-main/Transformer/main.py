import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import re
import time
import matplotlib.pyplot as plt
from datetime import datetime

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据预处理 
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. 定义模型 
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./Emotion_analysis-main/Transformer/bert-base-chinese')
        self.dropout = nn.Dropout(0.5)  # 增加Dropout比例，防止过拟合
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.linear(output)

# 3. 加载和预处理数据 
def load_data(file_path):
    texts = []
    labels = []
    
    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' not found!")
        return texts, labels
    
    # utf-8 (GB2312)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('####')
            if len(parts) < 2:
                continue
            label = parts[0].strip()
            text = '####'.join(parts[1:]).strip()
            
            clean_text = re.sub(r'\s+', ' ', text).strip()
            #  字符串过滤
            if 10 <= len(clean_text) <= 500:  
                try:
                    labels.append(int(label))
                    texts.append(clean_text)
                except ValueError:
                    print(f"Skipping invalid label: {label} | Text: {clean_text}")
    
    return texts, labels

# 4. 训练函数 
def train_model(model, data_loader, optimizer, criterion, device, epoch, total_epochs, scheduler=None):
    model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), 
                        desc=f"Epoch {epoch+1}/{total_epochs}", position=0, leave=True)
    
    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        losses.append(loss.item())
        batch_correct = torch.sum(preds == labels).item()
        correct_predictions += batch_correct
        total_samples += labels.size(0)
        
        avg_loss = np.mean(losses)
        accuracy = correct_predictions / total_samples
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })
    
    epoch_loss = np.mean(losses)
    epoch_accuracy = correct_predictions / len(data_loader.dataset)
    
    return epoch_loss, epoch_accuracy

# 5. 评估函数
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            correct_predictions += torch.sum(preds == labels).item()
    
    epoch_loss = np.mean(losses)
    epoch_accuracy = correct_predictions / len(data_loader.dataset)
    
    return epoch_loss, epoch_accuracy

# 主程序
if __name__ == "__main__":
    # 参数设置
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 3  # 3训练轮数
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01  # 添加权重衰减
    
    # 日志
    log_dir = "./Emotion_analysis-main/Transformer/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    data_file = "./Emotion_analysis-main/Transformer/data/comments.txt"
    print(f"Loading data from {data_file}...")
    texts, labels = load_data(data_file)
    
    if len(texts) == 0:
        print("No valid data found. Exiting.")
        exit(1)
        
    print(f"Loaded {len(texts)} samples")
    
    # 划分数据集 分层抽样
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # 检查本地模型
    local_model_path = './Emotion_analysis-main/Transformer/bert-base-chinese'
    if not os.path.exists(local_model_path):
        print(f"Error: Local model directory '{local_model_path}' not found!")
        exit(1)
    
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    
    # 创建数据集
    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = SentimentClassifier(n_classes=2).to(device)
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), 
                           lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)  # 添加权重衰减
    criterion = nn.CrossEntropyLoss()
    
    # 添加学习率调度器
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练统计
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_accuracy = 0
    early_stop_counter = 0
    PATIENCE = 2  # 早停耐心值
    
    print("\nStarting training with regularization...")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^8} | {'Val Loss':^10} | {'Val Acc':^8} | Status")
    print("-" * 70)
    
    for epoch in range(EPOCHS):
        # 训练阶段
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, criterion, device, epoch, EPOCHS, scheduler
        )
        
        # 验证阶段
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        
        # 记录统计信息
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印epoch总结
        print(f"{epoch+1:^6} | {train_loss:^10.4f} | {train_acc:^8.4f} | {val_loss:^10.4f} | {val_acc:^8.4f}", end=" | ")
        
        # best_model
        # 早停机制
        if val_acc > best_accuracy:
            best_model_path = f'./Emotion_analysis-main/Transformer/model/best_model_{timestamp}.bin'
            torch.save(model.state_dict(), best_model_path)
            best_accuracy = val_acc
            early_stop_counter = 0
            print(f"Saved best model: {best_model_path}")
        else:
            early_stop_counter += 1
            print("No improvement")
            
        if early_stop_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # final_model
    # 保存最终模型
    final_model_path = f'./Emotion_analysis-main/Transformer/model/final_model_{timestamp}.bin'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model: {final_model_path}")
    
    # 保存训练日志
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]},{train_accs[i]},{val_losses[i]},{val_accs[i]}\n")
    print(f"Saved training log: {log_file}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, 'b-o', label='Train')
    plt.plot(val_accs, 'r-o', label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, 'b-o', label='Train')
    plt.plot(val_losses, 'r-o', label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(log_dir, f"training_plot_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Saved training plot: {plot_file}")
    
    # 显示最终结果
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")