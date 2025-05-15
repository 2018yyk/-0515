import pandas as pd
import torch
from torch import nn
from diffusers import DDPMScheduler
import numpy as np
from models.unet1d_linear import ExtendedLinearUNet
import random

def get_time_encoding(times, dim=8):
    device = times.device
    pos = times.unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-np.log(10000.0) / dim))
    encoding = torch.zeros(times.size(0), dim, device=device)
    encoding[:, 0::2] = torch.sin(pos * div_term)
    encoding[:, 1::2] = torch.cos(pos * div_term)
    return encoding


# 定义时序分类器
class TimeOrderClassifier(nn.Module):
    def __init__(self, input_size):
        super(TimeOrderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)  # 输出为 2 类，表示先后顺序

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 训练扩散模型
def train_model(model, classifier, features_tensor, times_tensor, noise_scheduler, optimizer, num_epochs, guidance_weight=1.0):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        losses = []
        for i in range(len(features_tensor)):
            t = random.randint(0, noise_scheduler.config.num_train_timesteps)
            timesteps = torch.tensor([t], dtype=torch.long)

            # 向数据中添加噪声
            noisy_features = noise_scheduler.add_noise(features_tensor[i].unsqueeze(0),
                                                        torch.randn_like(features_tensor[i].unsqueeze(0)),
                                                        timesteps)
            
            time_encoding = get_time_encoding(times_tensor[i].unsqueeze(0))

            # 将时间编码和特征拼接加起来输入模型
            input_data = torch.add(noisy_features, time_encoding)
            # input_data = torch.cat((noisy_features, time_encoding), dim=1)          
            # 预测噪声
            noise_pred = model(input_data, timesteps)

            # 计算扩散模型的损失
            diffusion_loss = nn.functional.mse_loss(noise_pred,
                                                    torch.randn_like(features_tensor[i].unsqueeze(0)))

            # 随机选择一个样本进行拼接
            if i > 0 and i < len(features_tensor) - 1:
                # 有前有后，随机选一个
                if random.random() < 0.5:
                    # 选前面的样本
                    random_index = random.randint(0, i - 1)
                else:
                    # 选后面的样本
                    random_index = random.randint(i + 1, len(features_tensor) - 1)
            elif i == 0:
                # 第一个样本，只能选后面的
                random_index = random.randint(i + 1, len(features_tensor) - 1)
            else:
                # 最后一个样本，只能选前面的
                random_index = random.randint(0, i - 1)

            current_feature = features_tensor[i].unsqueeze(0)
            random_feature = features_tensor[random_index].unsqueeze(0)

            if random_index > i:
                combined_feature = torch.cat([current_feature, random_feature], dim=1)
                order_pred = classifier(combined_feature)
                true_order = torch.tensor([0], dtype=torch.long)
                classifier_loss_right = criterion(order_pred, true_order)

                combined_feature_wrong = torch.cat([random_feature, current_feature], dim=1)
                order_pred_wrong = classifier(combined_feature_wrong)
                true_order_wrong = torch.tensor([1], dtype=torch.long)  # 错误顺序标签为 1
                classifier_loss_wrong = criterion(order_pred_wrong, true_order_wrong)
            else:
                combined_feature = torch.cat([random_feature, current_feature], dim=1)
                order_pred = classifier(combined_feature)
                true_order = torch.tensor([0], dtype=torch.long)
                classifier_loss_right = criterion(order_pred, true_order)

                combined_feature_wrong = torch.cat([current_feature, random_feature], dim=1)
                order_pred_wrong = classifier(combined_feature_wrong)
                true_order_wrong = torch.tensor([1], dtype=torch.long)  # 错误顺序标签为 1
                classifier_loss_wrong = criterion(order_pred_wrong, true_order_wrong)

            classifier_loss = (classifier_loss_right + classifier_loss_wrong) / 2

            total_loss = diffusion_loss + guidance_weight * classifier_loss

            losses.append(total_loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {np.mean(losses)}')

# 验证扩散模型
def validate_model(model, classifier, val_features_tensor, val_times_tensor, noise_scheduler, guidance_weight=1.0):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    classifier.eval()
    losses = []
    with torch.no_grad():
        for i in range(len(val_features_tensor)):
            t = random.randint(0, noise_scheduler.config.num_train_timesteps)
            timesteps = torch.tensor([t], dtype=torch.long)

            # 向数据中添加噪声
            noisy_features = noise_scheduler.add_noise(val_features_tensor[i].unsqueeze(0),
                                                        torch.randn_like(val_features_tensor[i].unsqueeze(0)),
                                                        timesteps)

            time_encoding = get_time_encoding(val_times_tensor[i].unsqueeze(0))

            # 将时间编码和特征拼接加起来输入模型
            input_data = torch.add(noisy_features, time_encoding)
            # input_data = torch.cat((noisy_features, time_encoding), dim=1)          
            # 预测噪声
            noise_pred = model(input_data, timesteps)

            # 计算扩散模型的损失
            diffusion_loss = nn.functional.mse_loss(noise_pred,
                                                    torch.randn_like(val_features_tensor[i].unsqueeze(0)))
            # 随机选择一个样本进行拼接
            if i > 0 and i < len(val_features_tensor) - 1:
                # 有前有后，随机选一个
                if random.random() < 0.5:
                    # 选前面的样本
                    random_index = random.randint(0, i - 1)
                else:
                    # 选后面的样本
                    random_index = random.randint(i + 1, len(val_features_tensor) - 1)
            elif i == 0:
                # 第一个样本，只能选后面的
                random_index = random.randint(i + 1, len(val_features_tensor) - 1)
            else:
                # 最后一个样本，只能选前面的
                random_index = random.randint(0, i - 1)

            current_feature = val_features_tensor[i].unsqueeze(0)
            random_feature = val_features_tensor[random_index].unsqueeze(0)

            if random_index > i:
                combined_feature = torch.cat([current_feature, random_feature], dim=1)
                order_pred = classifier(combined_feature)
                true_order = torch.tensor([0], dtype=torch.long)
                classifier_loss_right = criterion(order_pred, true_order)

                combined_feature_wrong = torch.cat([random_feature, current_feature], dim=1)
                order_pred_wrong = classifier(combined_feature_wrong)
                true_order_wrong = torch.tensor([1], dtype=torch.long)  # 错误顺序标签为 1
                classifier_loss_wrong = criterion(order_pred_wrong, true_order_wrong)
            else:
                combined_feature = torch.cat([random_feature, current_feature], dim=1)
                order_pred = classifier(combined_feature)
                true_order = torch.tensor([0], dtype=torch.long)
                classifier_loss_right = criterion(order_pred, true_order)

                combined_feature_wrong = torch.cat([current_feature, random_feature], dim=1)
                order_pred_wrong = classifier(combined_feature_wrong)
                true_order_wrong = torch.tensor([1], dtype=torch.long)  # 错误顺序标签为 1
                classifier_loss_wrong = criterion(order_pred_wrong, true_order_wrong)

            classifier_loss = (classifier_loss_right + classifier_loss_wrong) / 2

            total_loss = diffusion_loss + guidance_weight * classifier_loss
            losses.append(total_loss.item())
    avg_loss = np.mean(losses)
    return avg_loss

def sample(model, classifier, features_tensor, times, noise_scheduler, features_mean, features_std):

    model.eval()
    classifier.eval()
    generated_data = []
    # 将时间张量转换为pandas的DatetimeIndex
    times = pd.to_datetime(times, unit='m')
    unique_dates = times.normalize().unique()  # 获取唯一的日期

    for date in unique_dates:
        for hour in range(24):
            start_time = date + pd.Timedelta(hours=hour)
            # end_time = start_time + pd.Timedelta(hours=1)

            # # 提取当前小时内的原始数据索引
            # hour_mask = (times >= start_time) & (times < end_time)
            # hour_indices = np.where(hour_mask)[0]
            # hour_features = features_tensor[hour_indices]

            # # 获取当前小时内原始数据的分钟级时间戳
            # hour_times_ns = times[hour_mask].values
            # if hour_times_ns.size > 0:
            #     hour_times_min = hour_times_ns // (60 * 10**9)
            # else:
            #     hour_times_min = np.array([], dtype=np.int64)



            # 生成当前小时内的60分钟所有，数据
            for minute in range(60):
                current_time = start_time + pd.Timedelta(minutes=minute)
                current_time_ns = current_time.value
                current_time_min = current_time_ns // (60 * 10**9)

                # # 处理插值（若有原始数据）
                # if len(hour_indices) > 0 and current_time_min in hour_times_min:
                #     # 原始分钟上有数据，直接使用该数据
                #     index = np.where(hour_times_min == current_time_min)[0][0]
                #     interpolated_features = hour_features[index].unsqueeze(0).float()
                #     # 时间编码
                #     current_time_tensor = torch.tensor([current_time_min], dtype=torch.long)
                #     time_encoding = get_time_encoding(current_time_tensor)
                #     interpolated_features = torch.add(interpolated_features, time_encoding)
                # else:
                #     # 原始分钟上无数据，使用默认时间编码与噪声拼接作为输入
                #     current_time_tensor = torch.tensor([current_time_min], dtype=torch.long)
                #     time_encoding = get_time_encoding(current_time_tensor)
                #     # 这里假设噪声形状与时间编码一致，你可以根据实际情况修改
                #     noise = torch.randn_like(time_encoding)
                #     interpolated_features = torch.add(time_encoding, noise)

                # 原始分钟无论有没有数据，使用默认时间编码与噪声拼接作为输入
                current_time_tensor = torch.tensor([current_time_min], dtype=torch.long)
                time_encoding = get_time_encoding(current_time_tensor)
                # 这里假设噪声形状与时间编码一致，你可以根据实际情况修改
                noise = torch.randn_like(time_encoding)
                sample = torch.add(time_encoding, noise)
                
                # 模型推理
                with torch.no_grad():
                    for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                        timesteps = torch.tensor([t], dtype=torch.long)
                        noise_pred = model(sample, timesteps)

                        # 分类器调整逻辑（保持不变）
                        if minute < 59:
                            next_time = start_time + pd.Timedelta(minutes=minute + 1)
                            next_time_ns = next_time.value
                            next_time_min = next_time_ns // (60 * 10**9)
                            next_time_tensor = torch.tensor([next_time_min], dtype=torch.long)
                            next_time_encoding = get_time_encoding(next_time_tensor)
                            next_noise = torch.randn_like(next_time_encoding)
                            next_interpolated_features = torch.add(next_time_encoding, next_noise)
                            next_noise_pred = model(next_interpolated_features, timesteps)                            

                            combined_sample = torch.cat([sample, next_interpolated_features], dim=1)
                            order_pred = classifier(combined_sample)
                            order_prob = torch.softmax(order_pred, dim=1)
                            if order_prob[0, 1] > 0.5:
                                temp = noise_pred.clone()
                                noise_pred = next_noise_pred
                                next_noise_pred = temp

                        sample = noise_scheduler.step(noise_pred, t, sample).prev_sample

                    sample = sample.squeeze(0).numpy()
                    sample = sample * features_std.numpy().squeeze() + features_mean.numpy().squeeze()

                # 保存生成时间（保持Timestamp格式，方便输出）
                generated_data.append([current_time, *sample])

    return generated_data


# 生成每分钟的数据
def generate_minute_data(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path, encoding='latin1')

    # 将 RECORD_TIME 列转换为日期时间类型
    df['RECORD_TIME'] = pd.to_datetime(df['RECORD_TIME'])

    # 按照 RECORD_TIME 列进行排序
    df = df.sort_values(by='RECORD_TIME')

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)

    # 提取需要参与训练的特征列
    feature_columns = ['TXTSI', 'TXTS', 'TXTP', 'TXTTI', 'TXTC', 'TXTV', 'TXTCR', 'TXTMN']
    features = df[feature_columns].values.astype(np.float32)

    # 提取时间列
    times = df['RECORD_TIME'].values.astype('datetime64[m]').astype(np.int64)

    # 将数据转换为 PyTorch 张量
    features_tensor = torch.from_numpy(features)
    times_tensor = torch.from_numpy(times)

    # 归一化特征数据
    features_mean = features_tensor.mean(dim=0, keepdim=True)
    features_std = features_tensor.std(dim=0, keepdim=True)
    features_tensor = (features_tensor - features_mean) / features_std

    # 划分训练集和验证集（简单按 80 - 20 划分）
    train_size = int(0.8 * len(features_tensor))
    train_features_tensor = features_tensor[:train_size]
    train_times_tensor = times_tensor[:train_size]
    val_features_tensor = features_tensor[train_size:]
    val_times_tensor = times_tensor[train_size:]

    # 创建扩散模型调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 创建扩散模型和分类器
    model = ExtendedLinearUNet()
    classifier = TimeOrderClassifier(features_tensor.shape[1] * 2)

    # 定义优化器
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)

    # 训练模型
    num_epochs = 50
    train_model(model, classifier, train_features_tensor, train_times_tensor, noise_scheduler, optimizer, num_epochs)

    # 验证模型
    val_loss = validate_model(model, classifier, val_features_tensor, val_times_tensor, noise_scheduler)
    print(f'Validation Loss: {val_loss}')

    syn_data = sample(model=model, classifier=classifier, features_tensor=features_tensor, times=times, noise_scheduler=noise_scheduler, features_mean=features_mean, features_std=features_std)

    all_columns = ['RECORD_TIME', 'TXTSI', 'TXTS', 'TXTP', 'TXTTI', 'TXTC', 'TXTV', 'TXTCR', 'TXTMN']
    df = pd.DataFrame(syn_data, columns=all_columns)
    df.to_csv('generated_data.csv', index=False)

if __name__ == '__main__':
    file_path = 'E:\diffusion_gaolu-master-zy515\高炉运行参数.xlsx'
    generate_minute_data(file_path)