"""
数据转换脚本：将 cascades 格式数据转换为 RecBole 所需格式
"""
import os
import pandas as pd
from datetime import datetime

def parse_cascades_file(filepath):
    """解析 cascades 格式文件"""
    data = []
    user_id = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 每行代表一个用户的交互序列
            interactions = line.split(',')
            for interaction in interactions:
                interaction = interaction.strip()
                if not interaction:
                    continue
                
                parts = interaction.split()
                if len(parts) >= 3:
                    item_id = parts[0]
                    timestamp = parts[1]
                    correctness = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    data.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'timestamp': timestamp,
                        'correctness': correctness
                    })
            
            user_id += 1
    
    return pd.DataFrame(data)

def convert_to_recbole_format(df, dataset_name):
    """转换为 RecBole 格式"""
    # 创建数据集目录
    dataset_dir = f'dataset/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 转换时间戳格式（假设原始时间戳为 YYMMDDHHMM 格式）
    def convert_timestamp(ts):
        try:
            # 处理不同长度的时间戳
            ts_str = str(ts)
            if len(ts_str) == 10:  # YYMMDDHHMM
                dt = datetime.strptime(ts_str, '%y%m%d%H%M')
                return int(dt.timestamp())
            elif len(ts_str) == 11:  # 可能是其他格式
                dt = datetime.strptime(ts_str[:10], '%y%m%d%H%M')
                return int(dt.timestamp())
            else:
                return 0
        except:
            return 0
    
    df['timestamp'] = df['timestamp'].apply(convert_timestamp)
    
    # 生成 RecBole 所需文件
    # 1. 用户交互文件 (.inter)
    inter_df = df[['user_id', 'item_id', 'timestamp', 'correctness']].copy()
    inter_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float', 'correctness:float']
    inter_df.to_csv(f'{dataset_dir}/{dataset_name}.inter', sep='\t', index=False)
    
    # 2. 用户文件 (.user)
    user_ids = df['user_id'].unique()
    user_df = pd.DataFrame({'user_id:token': user_ids})
    user_df.to_csv(f'{dataset_dir}/{dataset_name}.user', sep='\t', index=False)
    
    # 3. 物品文件 (.item)
    item_ids = df['item_id'].unique()
    item_df = pd.DataFrame({'item_id:token': item_ids})
    item_df.to_csv(f'{dataset_dir}/{dataset_name}.item', sep='\t', index=False)
    
    print(f"数据集 {dataset_name} 转换完成:")
    print(f"  - 用户数: {len(user_ids)}")
    print(f"  - 物品数: {len(item_ids)}")
    print(f"  - 交互数: {len(df)}")
    print(f"  - 文件保存在: {dataset_dir}/")
    
    return dataset_dir

def main():
    # 转换 MOOPer 数据集
    print("处理 cascades_MOOPer.txt...")
    df_mooper = parse_cascades_file('cascades_MOOPer.txt')
    convert_to_recbole_format(df_mooper, 'mooper')
    
    print("\n处理 cascades_Assist09.txt...")
    df_assist = parse_cascades_file('cascades_Assist09.txt')
    convert_to_recbole_format(df_assist, 'assist09')

if __name__ == '__main__':
    main()
