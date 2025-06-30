import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDPMScheduler

from model import ExtendedLinearUNet, TimeOrderClassifier, TimeEmbedding
from train_utils import train_model, validate_model
from sample_utils import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_minute_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df['作业时间'] = pd.to_datetime(df['作业时间'])
    df = df.sort_values(by='作业时间')

    feature_columns = ['小时料批', '风量', '冷风压力', '热风压力', '炉顶压力', '压差', '透气性指数', '富氧量', '富氧率', '热风温度',
                       '喷煤量', '煤比', '燃料比', '顶温最高', '顶温最低', '顶温平均', '标准风速', '实际风速', 'K值',
                       '鼓风湿度', '鼓风动能', '理论燃烧温度', '煤气利用率', '炉腹煤气量', '全炉温差']
    features = df[feature_columns].values.astype(np.float32)
    times = df['作业时间'].values.astype('datetime64[ns]')
    start_time = np.array([pd.to_datetime(times[0].astype('datetime64[D]'))] * len(times), dtype='datetime64[ns]')
    times_min = (times - start_time) // pd.Timedelta(minutes=1)

    times_tensor = torch.from_numpy(times_min.astype(np.int64)).to(device)
    features_tensor = torch.from_numpy(features).to(device)
    features_mean = features_tensor.mean(dim=0, keepdim=True).to(device)
    features_std = features_tensor.std(dim=0, keepdim=True).to(device)
    features_tensor = (features_tensor - features_mean) / features_std

    train_size = int(0.8 * len(features_tensor))
    train_features, val_features = features_tensor[:train_size], features_tensor[train_size:]
    train_times, val_times = times_tensor[:train_size], times_tensor[train_size:]

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = ExtendedLinearUNet().to(device)
    classifier = TimeOrderClassifier(input_size=features.shape[1] * 2).to(device)
    time_embed = TimeEmbedding(max_time=1000000, embedding_dim=64, output_dim=128).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()) + list(time_embed.parameters()), lr=1e-4)

    epoch_losses = train_model(model, classifier, train_features, train_times, time_embed, noise_scheduler, optimizer, num_epochs=1000)
    val_loss = validate_model(model, classifier, val_features, val_times, time_embed, noise_scheduler)
    print(f'Validation Loss: {val_loss:.4f}')

    syn_data = sample(model, classifier, times, noise_scheduler, features_mean, features_std)

    flat_syn_data = [[row[0]] + row[1] for row in syn_data]
    generated_df = pd.DataFrame(flat_syn_data, columns=['作业时间'] + feature_columns)
    generated_df.to_csv('generated_minute_data.csv', index=False, encoding='utf-8-sig')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    file_path = 'data/高炉运行参数.xlsx'
    generate_minute_data(file_path)
