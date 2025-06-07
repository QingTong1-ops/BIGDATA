import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import os
import csv
from datetime import datetime

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义模型（与训练程序相同）
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifier, self).__init__()
        # 确保本地模型目录存在
        if not os.path.exists('./Emotion_analysis-main/Transformer/bert-base-chinese'):
            print("Error: 'bert-base-chinese' directory not found!")
            print("Please download the Chinese BERT model and place it in the current directory.")
            exit(1)
            
        self.bert = BertModel.from_pretrained('./Emotion_analysis-main/Transformer/bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.linear(output)

# 预测函数
def predict_sentiment(text, model, tokenizer, max_len=128):
    model.eval()
    
    # 清理文本
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # 编码文本
    encoding = tokenizer.encode_plus(
        clean_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)
    
    # 获取置信度
    confidence = probs[0][prediction].item()
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    
    return sentiment, confidence, probs[0][1].item() if sentiment == "Positive" else probs[0][0].item()

def main():
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('./Emotion_analysis-main/Transformer/bert-base-chinese')
    
    # 加载模型
    model_path = './Emotion_analysis-main/Transformer/model/final_model_20250607_153032.bin'  # 训练好的模型文件
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    model = SentimentClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # 选择预测模式
    print("\n选择预测模式:")
    print("1. 使用示例文本预测")
    print("2. 读取test.txt文件进行批量预测")
    choice = input("请输入选择 (1 或 2): ")
    
    if choice == '1':
        # 使用两个示例文本
        test_samples = [
            "这部电影真的太精彩了，演员演技在线，剧情扣人心弦，强烈推荐！",
            "非常失望，剧情拖沓，特效也很假，浪费了我两个小时的时间。",
            "我不喜欢。",
            " UP好棒"
        ]
        
        print("\n预测结果:")
        print("-" * 80)
        for text in test_samples:
            sentiment, confidence, class_prob = predict_sentiment(text, model, tokenizer)
            print(f"文本: {text}")
            print(f"情感: {sentiment} | 置信度: {confidence:.4f} | 积极概率: {class_prob:.4f}")
            print("-" * 80)
    
    elif choice == '2':
        # 读取test.txt文件
        test_file = './Emotion_analysis-main/Transformer/data/test.txt'
        
        if not os.path.exists(test_file):
            print(f"Error: File '{test_file}' not found!")
            return
        
        with open(test_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        if not texts:
            print(f"Error: No valid texts found in '{test_file}'!")
            return
        
        print(f"\n开始预测 {len(texts)} 条评论...")
        
        results = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"处理中: {i+1}/{len(texts)}")
            
            sentiment, confidence, class_prob = predict_sentiment(text, model, tokenizer)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_prob': class_prob if sentiment == "Positive" else 1 - class_prob,
                'negative_prob': 1 - class_prob if sentiment == "Positive" else class_prob
            })
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'./Emotion_analysis-main/Transformer/data/predictions.csv'
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['text', 'sentiment', 'confidence', 'positive_prob', 'negative_prob']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n预测完成! 结果已保存至: {output_file}")
        
        # 显示统计信息
        positive_count = sum(1 for r in results if r['sentiment'] == "Positive")
        negative_count = len(results) - positive_count
        
        print("\n预测统计:")
        print(f"积极评论: {positive_count} ({positive_count/len(results)*100:.1f}%)")
        print(f"消极评论: {negative_count} ({negative_count/len(results)*100:.1f}%)")
        
        # 显示前5条结果
        print("\n前5条预测结果:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result['text'][:60]}{'...' if len(result['text']) > 60 else ''}")
            print(f"   情感: {result['sentiment']}, 置信度: {result['confidence']:.4f}")
    
    else:
        print("无效选择，请重新运行程序并输入1或2")

if __name__ == "__main__":
    main()