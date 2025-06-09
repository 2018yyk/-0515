import pandas as pd
import torch
from torch import nn
from diffusers import DDPMScheduler
import numpy as np
from models.unet1d_linear_for_gaolu import ExtendedLinearUNet  # 假设该模型已正确定义
import random
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def get_time_encoding(times, dim=25, max_time=10000000):
#
#     #  使用nn.Embedding
#     embedding = nn.Embedding(max_time + 1, dim)
#     encoded_tensor = embedding(times.to(device))
#     return encoded_tensor
embedding_dim = 25
max_time = 10000000
time_embedding = nn.Embedding(max_time + 1, embedding_dim).to(device)

def get_time_encoding(times):
    return time_embedding(times.to(device))

class TimeOrderClassifier(nn.Module):
    def __init__(self, input_size):
        super(TimeOrderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # 输出为2类：正确顺序（0）或错误顺序（1）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train_model(model, classifier, features_tensor, times_tensor, noise_scheduler, optimizer, num_epochs, guidance_weight=1.0):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        losses = []
        # print("Feature Tensor on:", features_tensor.device)
        # print("Time Tensor on:", times_tensor.device)
        # print("Embedding on:", next(time_embedding.parameters()).device)
        for i in range(len(features_tensor)):
            t = random.randint(0, noise_scheduler.config.num_train_timesteps-1)
            timesteps = torch.tensor([t], dtype=torch.long).to(device)
            
            # 添加噪声
            noisy_features = noise_scheduler.add_noise(
                features_tensor[i].unsqueeze(0),
                torch.randn_like(features_tensor[i].unsqueeze(0)),
                timesteps
            )
            time_encoding = get_time_encoding(times_tensor[i].unsqueeze(0))
            input_data = torch.add(noisy_features, time_encoding)
            
            # 扩散模型损失
            noise_pred = model(input_data, timesteps)
            # print("input_data-----:", input_data.device)
            # print("timesteps:----", timesteps.device)
            diffusion_loss = nn.functional.mse_loss(noise_pred, torch.randn_like(features_tensor[i].unsqueeze(0)))
            
            # 时序分类器损失
            if i > 0 and i < len(features_tensor) - 1:
                # 随机选择前后样本
                random_index = i + 1 if random.random() < 0.5 else i - 1
            elif i == 0:
                random_index = i + 1
            else:
                random_index = i - 1
            
            current_feature = features_tensor[i].unsqueeze(0)
            random_feature = features_tensor[random_index].unsqueeze(0)
            combined = torch.cat([current_feature, random_feature], dim=1)
            order_label = torch.tensor([0 if random_index > i else 1], dtype=torch.long).to(device)
            
            order_pred = classifier(combined)
            classifier_loss = criterion(order_pred, order_label)
            
            total_loss = diffusion_loss + guidance_weight * classifier_loss
            losses.append(total_loss.item())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {np.mean(losses):.4f}')

def validate_model(model, classifier, val_features_tensor, val_times_tensor, noise_scheduler, guidance_weight=1.0):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    classifier.eval()
    losses = []
    with torch.no_grad():
        for i in range(len(val_features_tensor)):
            t = random.randint(0, noise_scheduler.config.num_train_timesteps)
            timesteps = torch.tensor([t], dtype=torch.long).to(device)
            
            noisy_features = noise_scheduler.add_noise(
                val_features_tensor[i].unsqueeze(0),
                torch.randn_like(val_features_tensor[i].unsqueeze(0)),
                timesteps
            )
            time_encoding = get_time_encoding(val_times_tensor[i].unsqueeze(0))
            input_data = torch.add(noisy_features, time_encoding)
            
            noise_pred = model(input_data, timesteps)
            diffusion_loss = nn.functional.mse_loss(noise_pred, torch.randn_like(val_features_tensor[i].unsqueeze(0)))
            
            # 分类器损失计算（同上）
            if i > 0 and i < len(val_features_tensor) - 1:
                random_index = i + 1 if random.random() < 0.5 else i - 1
            elif i == 0:
                random_index = i + 1
            else:
                random_index = i - 1
            
            current_feature = val_features_tensor[i].unsqueeze(0)
            random_feature = val_features_tensor[random_index].unsqueeze(0)
            combined = torch.cat([current_feature, random_feature], dim=1)
            order_label = torch.tensor([0 if random_index > i else 1], dtype=torch.long).to(device)
            order_pred = classifier(combined)
            classifier_loss = criterion(order_pred, order_label)
            
            total_loss = diffusion_loss + guidance_weight * classifier_loss
            losses.append(total_loss.item())
    
    return np.mean(losses)

def sample(model, classifier, times, noise_scheduler, features_mean, features_std):
    model.eval()
    classifier.eval()
    generated_data = []
    times = pd.to_datetime(times, unit='ns')  # 转换为datetime对象
    times_series = pd.Series(times)

    # 生成所有日期的每小时每分钟数据
    for date in times_series.dt.normalize().unique():
        for hour in range(24):
            for minute in range(60):
                current_time = date + pd.Timedelta(hours=hour, minutes=minute)
                current_min = hour * 60 + minute  # 转换为总分钟数
                time_tensor = torch.tensor([current_min], dtype=torch.long).to(device)
                time_encoding = get_time_encoding(time_tensor)
                
                # 初始化噪声
                sample = torch.randn_like(time_encoding) * features_std + features_mean  # 逆归一化初始化？
                sample = (sample - features_mean) / features_std  # 重新归一化（确保输入范围正确）
                
                # 扩散模型逆过程
                for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                    timesteps = torch.tensor([t], dtype=torch.long).to(device)
                    input_data = torch.add(sample, time_encoding)
                    noise_pred = model(input_data, timesteps)
                    
                    # 分类器引导（强制时间顺序）
                    if minute < 59:
                        next_min = current_min + 1
                        next_time_encoding = get_time_encoding(torch.tensor([next_min], dtype=torch.long).to(device))
                        next_input = torch.add(sample, next_time_encoding)
                        next_noise_pred = model(next_input, timesteps)
                        
                        # 拼接当前和下一分钟特征，预测顺序
                        combined = torch.cat([sample, next_noise_pred], dim=1)
                        order_prob = torch.softmax(classifier(combined), dim=1)
                        
                        # 若预测下一分钟在后，交换噪声预测（确保顺序）
                        if order_prob[0, 1] > 0.5:
                            noise_pred, next_noise_pred = next_noise_pred, noise_pred
                    
                    sample = noise_scheduler.step(noise_pred, t, sample.unsqueeze(0)).prev_sample.squeeze(0).to(device)
                
                # 逆归一化
                # sample = sample.numpy() * features_std.numpy().squeeze() + features_mean.numpy().squeeze().to(device)
                sample = sample.detach().cpu().numpy() * features_std.detach().cpu().numpy().squeeze() + features_mean.detach().cpu().numpy().squeeze()

                generated_data.append([current_time] + sample.tolist())
                progress_bar.update(1)
    progress_bar.close()
    return generated_data


def generate_minute_data(file_path):
    # 读取数据（假设原始数据列为'作业时间'和特征列）
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df['作业时间'] = pd.to_datetime(df['作业时间'])
    df = df.sort_values(by='作业时间')
    
    # 提取特征列（需根据实际列名修改）
    feature_columns = ['小时料批', '风量', '冷风压力', '热风压力', '炉顶压力', '压差', '透气性指数', '富氧量', '富氧率', '热风温度', '喷煤量', '煤比', '燃料比', '顶温最高', '顶温最低', '顶温平均', '标准风速', '实际风速', 'K值', '鼓风湿度', '鼓风动能', '理论燃烧温度', '煤气利用率', '炉腹煤气量', '全炉温差']
    features = df[feature_columns].values.astype(np.float32)
    times = df['作业时间'].values.astype('datetime64[ns]')
    
    start_time = np.array([pd.to_datetime(times[0].astype('datetime64[D]'))] * len(times), dtype='datetime64[ns]')

    # 转换为分钟级时间戳（总分钟数）
    times_min = (times - start_time) // pd.Timedelta(minutes=1)
    times_tensor = torch.from_numpy(times_min.astype(np.int64)).to(device)
    features_tensor = torch.from_numpy(features).to(device)

    # 归一化
    features_mean = features_tensor.mean(dim=0, keepdim=True).to(device)
    features_std = features_tensor.std(dim=0, keepdim=True).to(device)
    features_tensor = (features_tensor - features_mean) / features_std
    
    # 划分数据集
    train_size = int(0.8 * len(features_tensor))
    train_features, val_features = features_tensor[:train_size], features_tensor[train_size:]
    train_times, val_times = times_tensor[:train_size], times_tensor[train_size:]
    
    # 初始化模型和调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = ExtendedLinearUNet().to(device)  # 确保模型输入维度正确（特征数+时间编码维度）
    classifier = TimeOrderClassifier(input_size=features.shape[1] * 2).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
    # print("Model on:", next(model.parameters()).device)
    # print("Classifier on:", next(classifier.parameters()).device)
    # # 训练
    train_model(model, classifier, train_features, train_times, noise_scheduler, optimizer, num_epochs=1)

    # 验证
    val_loss = validate_model(model, classifier, val_features, val_times, noise_scheduler)
    print(f'Validation Loss: {val_loss:.4f}')
    
    # 生成数据
    syn_data = sample(model, classifier, times, noise_scheduler, features_mean, features_std)

    # 保存结果
    all_columns = ['RECORD_TIME'] + feature_columns
    print(f"all_columns 数量: {len(all_columns)}")

    generated_df = pd.DataFrame(syn_data, columns=all_columns)
    generated_df.to_csv('generated_minute_data.csv', index=False)

if __name__ == '__main__':
    file_path = 'data/高炉运行参数.xlsx'
    # file_path = 'data/高炉运行参数-24-28号.xlsx'
    generate_minute_data(file_path)