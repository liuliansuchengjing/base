# RecBole 配置文件示例
# 保存为 .yaml 文件并在运行时指定

# 数据集配置
dataset: mooper
data_path: ./dataset/mooper/

# 评估配置
eval_args:
  split: 
    RS: [0.8, 0.1, 0.1]
  group_by: user
  order: TO
  mode: labeled

metrics: ['Hit', 'NDCG', 'MRR']
topk: [5, 10, 20]
valid_metric: Hit@20

# 训练配置
epochs: 50
train_batch_size: 1024
eval_batch_size: 1024
learning_rate: 0.001

# 设备配置
device: cpu

# 早停
early_stop: 5

# 日志
show_progress: true
verbose: true
