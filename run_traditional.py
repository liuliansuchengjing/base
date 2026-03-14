"""
传统推荐算法实现 - 不依赖 PyTorch
使用简单的基于统计的方法
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle


class SimpleRecommender:
    """简单的推荐基类"""
    def __init__(self):
        self.user_items = defaultdict(list)
        self.item_users = defaultdict(set)
        self.item_popularity = defaultdict(int)
        
    def fit(self, train_data):
        """训练"""
        for user, item, timestamp in train_data:
            self.user_items[user].append((item, timestamp))
            self.item_users[item].add(user)
            self.item_popularity[item] += 1
    
    def predict(self, user, items):
        """预测分数"""
        raise NotImplementedError


class PopRecommender(SimpleRecommender):
    """基于流行度的推荐"""
    def predict(self, user, items):
        scores = {item: self.item_popularity[item] for item in items}
        return scores


class ItemCFRecommender(SimpleRecommender):
    """基于物品的协同过滤"""
    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.item_similarity = {}
    
    def fit(self, train_data):
        """训练：计算物品相似度"""
        super().fit(train_data)
        
        print("计算物品相似度...")
        # 构建物品共现矩阵
        item_pairs = defaultdict(int)
        for user, items in self.user_items.items():
            items_list = [item for item, _ in items]
            for i in range(len(items_list)):
                for j in range(i+1, len(items_list)):
                    item_pairs[(items_list[i], items_list[j])] += 1
                    item_pairs[(items_list[j], items_list[i])] += 1
        
        # 计算余弦相似度
        for (item1, item2), count in item_pairs.items():
            sim = count / np.sqrt(self.item_popularity[item1] * self.item_popularity[item2])
            if item1 not in self.item_similarity:
                self.item_similarity[item1] = {}
            self.item_similarity[item1][item2] = sim
    
    def predict(self, user, items):
        """预测分数"""
        if user not in self.user_items:
            return {item: 0 for item in items}
        
        user_history = [item for item, _ in self.user_items[user]]
        scores = {}
        
        for item in items:
            score = 0
            if item in self.item_similarity:
                # 找最相似的 k 个物品
                similar_items = sorted(self.item_similarity[item].items(), 
                                      key=lambda x: x[1], reverse=True)[:self.k]
                for sim_item, sim in similar_items:
                    if sim_item in user_history:
                        score += sim
            scores[item] = score
        
        return scores


def load_data(dataset_path):
    """加载 RecBole 格式的数据"""
    inter_file = os.path.join(dataset_path, f'{os.path.basename(dataset_path)}.inter')
    df = pd.read_csv(inter_file, sep='\t')
    
    # 重命名列
    df.columns = ['user_id', 'item_id', 'timestamp', 'correctness']
    
    # 按时间排序
    df = df.sort_values(['user_id', 'timestamp'])
    
    return df


def split_data(df, test_ratio=0.2):
    """划分训练集和测试集"""
    train_data = []
    test_data = {}
    
    for user in df['user_id'].unique():
        user_df = df[df['user_id'] == user]
        items = user_df['item_id'].tolist()
        timestamps = user_df['timestamp'].tolist()
        
        # 按时间划分
        split_idx = int(len(items) * (1 - test_ratio))
        
        # 训练数据
        for i in range(split_idx):
            train_data.append((user, items[i], timestamps[i]))
        
        # 测试数据：最后一个物品作为测试目标，前面作为历史
        if split_idx < len(items):
            test_data[user] = {
                'history': items[:split_idx],
                'target': items[split_idx:],
                'timestamps': timestamps[split_idx:]
            }
    
    return train_data, test_data


def evaluate(recommender, test_data, all_items, k_list=[5, 10, 20]):
    """评估推荐效果"""
    metrics = {k: {'Hit': [], 'NDCG': [], 'MRR': [], 'AP': []} for k in k_list}
    
    total = len(test_data)
    processed = 0
    
    for user, data in test_data.items():
        history = data['history']
        targets = data['target']
        
        if not targets:
            continue
        
        # 候选物品：所有物品 - 历史物品
        candidates = list(all_items - set(history))
        
        # 预测分数
        scores = recommender.predict(user, candidates)
        
        # 排序
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_items = [item for item, _ in ranked_items]
        
        # 计算指标
        for k in k_list:
            top_k = ranked_items[:k]
            
            # Hit
            hit = any(item in top_k for item in targets)
            metrics[k]['Hit'].append(1 if hit else 0)
            
            # NDCG
            dcg = 0
            for i, item in enumerate(top_k):
                if item in targets:
                    dcg += 1 / np.log2(i + 2)
            
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(targets), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[k]['NDCG'].append(ndcg)
            
            # MRR
            mrr = 0
            for i, item in enumerate(top_k):
                if item in targets:
                    mrr = 1 / (i + 1)
                    break
            metrics[k]['MRR'].append(mrr)
            
            # MAP (Mean Average Precision)
            # 计算单个用户的 AP@K
            ap_sum = 0
            hit_count = 0
            for i, item in enumerate(top_k):
                if item in targets:
                    hit_count += 1
                    precision_at_i = hit_count / (i + 1)
                    ap_sum += precision_at_i
            # AP = sum of precision at each relevant item / min(K, total relevant items)
            ap = ap_sum / min(k, len(targets)) if len(targets) > 0 else 0
            metrics[k]['AP'].append(ap)
        
        processed += 1
        if processed % 100 == 0:
            print(f"评估进度: {processed}/{total}")
    
    # 计算平均值
    results = {}
    for k in k_list:
        results[f'Hit@{k}'] = np.mean(metrics[k]['Hit'])
        results[f'NDCG@{k}'] = np.mean(metrics[k]['NDCG'])
        results[f'MRR@{k}'] = np.mean(metrics[k]['MRR'])
        results[f'MAP@{k}'] = np.mean(metrics[k]['AP'])  # 添加 MAP@K
    
    return results


def run_experiment(dataset_name):
    """运行单个数据集实验"""
    print(f'\n{"="*80}')
    print(f'运行实验: {dataset_name}')
    print(f'{"="*80}\n')
    
    # 加载数据
    print("加载数据...")
    dataset_path = f'./dataset/{dataset_name}'
    df = load_data(dataset_path)
    
    all_items = set(df['item_id'].unique())
    user_count = df['user_id'].nunique()
    print(f"总交互数: {len(df)}")
    print(f"用户数: {user_count}")  
    print(f"物品数: {len(all_items)}")
    
    # 划分数据
    print("\n划分数据...")
    train_data, test_data = split_data(df)
    print(f"训练样本: {len(train_data)}")
    print(f"测试用户: {len(test_data)}")
    
    results = {}
    models = {}  # 保存训练好的模型
    
    # 1. Popularity
    print("\n训练 Popularity 模型...")
    pop = PopRecommender()
    pop.fit(train_data)
    print("评估 Popularity...")
    pop_results = evaluate(pop, test_data, all_items)
    results['Pop'] = pop_results
    models['Pop'] = pop  # 保存模型
    print(f"Pop 结果: {pop_results}")
    
    # 2. ItemCF
    print("\n训练 ItemCF 模型...")
    itemcf = ItemCFRecommender(k=10)
    itemcf.fit(train_data)
    print("评估 ItemCF...")
    itemcf_results = evaluate(itemcf, test_data, all_items)
    results['ItemCF'] = itemcf_results
    models['ItemCF'] = itemcf  # 保存模型
    print(f"ItemCF 结果: {itemcf_results}")
    
    return results, models


def main():
    """主函数 - 运行 mooper 和 assist09 数据集"""
    datasets = ['mooper', 'assist09']  # 运行两个数据集
    
    all_results = {}
    all_models = {}
    
    for dataset in datasets:
        try:
            results, models = run_experiment(dataset)
            all_results[dataset] = results
            all_models[dataset] = models
        except Exception as e:
            print(f"\n错误: {dataset} 运行失败")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    print('\n' + '='*80)
    print('所有数据集实验结果汇总')
    print('='*80)
    
    for dataset, models_results in all_results.items():
        print(f'\n数据集: {dataset}')
        print('-'*80)
        for model, metrics in models_results.items():
            print(f'\n  模型: {model}')
            for metric, value in metrics.items():
                print(f'    {metric}: {value:.4f}')
    
    # 保存评估结果
    with open('results_all.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print('\n评估结果已保存到 results_all.pkl')
    
    # 保存模型权重
    print('\n保存模型权重...')
    os.makedirs('./saved_models', exist_ok=True)
    for dataset, models in all_models.items():
        for model_name, model in models.items():
            model_path = f'./saved_models/{dataset}_{model_name.lower()}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f'  {model_name} 模型已保存到 {model_path}')
    
    print('\n所有模型权重保存完成！')


if __name__ == '__main__':
    main()
