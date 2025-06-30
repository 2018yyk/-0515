import torch
import pandas as pd
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 25
max_time = 10000000
time_embedding = nn.Embedding(max_time + 1, embedding_dim).to(device)

def get_time_encoding(times):
    return time_embedding(times.to(device))
import torch
import pandas as pd
from tqdm import tqdm
# from your_model_file import get_time_encoding  # 请替换为你实际的模块名

@torch.no_grad()
def sample(model, classifier, times, noise_scheduler, features_mean, features_std,
           inference_steps=50, classifier_guidance_every=10):
    """
    带分类器引导的高效采样函数
    参数：
        model: 扩散模型
        classifier: 时序分类器
        times: 时间序列 (np.datetime64数组)
        noise_scheduler: DDPMScheduler实例
        features_mean, features_std: 归一化参数，tensor形状应为 (1, 25)
        inference_steps: 逆扩散步数
        classifier_guidance_every: 每隔多少步执行分类器引导
    """

    model.eval()
    classifier.eval()
    device = features_mean.device

    times = pd.to_datetime(times, unit='ns')
    times_series = pd.Series(times)

    total_iterations = len(times_series.dt.normalize().unique()) * 24 * 60
    progress_bar = tqdm(total=total_iterations, desc="生成数据进度")

    noise_scheduler.set_timesteps(inference_steps)

    generated_data = []

    feature_dim = features_mean.shape[1]  # 应该是25

    for date in times_series.dt.normalize().unique():
        for hour in range(24):
            for minute in range(60):
                current_time = date + pd.Timedelta(hours=hour, minutes=minute)
                current_min = hour * 60 + minute

                time_tensor = torch.tensor([current_min], dtype=torch.long, device=device)
                time_encoding = get_time_encoding(time_tensor)  # [1, 25]

                # 初始化采样噪声，shape为 (1, feature_dim)
                sample = torch.randn(1, feature_dim, device=device) * features_std + features_mean
                sample = (sample - features_mean) / features_std  # 归一化后输入模型

                for i, t in enumerate(reversed(noise_scheduler.timesteps)):
                    t_tensor = torch.tensor([t], dtype=torch.long, device=device)
                    input_data = sample + time_encoding  # 形状匹配 (1, 25)

                    noise_pred = model(input_data, t_tensor)

                    # 分类器引导
                    if minute < 59 and (i % classifier_guidance_every == 0 or i < 5):
                        next_min = current_min + 1
                        next_time_encoding = get_time_encoding(torch.tensor([next_min], dtype=torch.long, device=device))
                        next_input = sample + next_time_encoding
                        next_noise_pred = model(next_input, t_tensor)

                        combined = torch.cat([sample, next_noise_pred], dim=1)  # (1, 50)
                        order_prob = torch.softmax(classifier(combined), dim=1)

                        if order_prob[0, 1] > 0.5:
                            noise_pred, next_noise_pred = next_noise_pred, noise_pred

                    sample = noise_scheduler.step(noise_pred, t, sample.unsqueeze(0)).prev_sample.squeeze(0).to(device)

                # 逆归一化
                sample = sample.detach().cpu().numpy() * features_std.cpu().numpy().squeeze() + features_mean.cpu().numpy().squeeze()

                generated_data.append([current_time] + sample.tolist())
                progress_bar.update(1)

    progress_bar.close()
    return generated_data
