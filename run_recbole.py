"""
使用 RecBole 运行序列推荐模型
支持的模型：SASRec, GRU4Rec, SR-GNN
"""
import os
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer, set_color
from recbole.data.interaction import Interaction


def run_model(model_name, dataset_name, config_dict=None):
    """运行单个模型"""
    print(set_color(f'\n========== 运行 {model_name} 在 {dataset_name} 数据集 ==========', 'green'))
    
    # 基础配置
    base_config = {
        'model': model_name,
        'dataset': dataset_name,
        'data_path': f'./dataset/{dataset_name}',
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},  # 训练集、验证集、测试集比例
            'group_by': 'user',
            'order': 'TO',
            'mode': 'labeled',
        },
        'metrics': ['Hit', 'NDCG', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'Hit@20',
        'learning_rate': 0.001,
        'training_neg_sample_num': 1,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'epochs': 100,
        'early_stop': 10,  # 早停轮数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 2020,
        'show_progress': True,
        'verbose': True,
    }
    
    # 合并自定义配置
    if config_dict:
        base_config.update(config_dict)
    
    # 初始化配置
    config = Config(model=model_name, dataset=dataset_name, config_dict=base_config)
    init_seed(config['seed'], config['reproducibility'])
    
    # 创建数据集
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 初始化模型
    model = get_model(config['model'])(config, dataset).to(config['device'])
    
    # 训练器
    trainer = get_trainer(config['model_type'])(config, model)
    
    # 训练
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    # 测试
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    
    print(set_color(f'\n{model_name} 在 {dataset_name} 的结果:', 'yellow'))
    print(set_color(f'验证集最佳结果: {best_valid_result}', 'yellow'))
    print(set_color(f'测试集结果: {test_result}', 'yellow'))
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'valid_result': best_valid_result,
        'test_result': test_result
    }


def run_all_models(datasets=['mooper', 'assist09'], models=['SASRec', 'GRU4Rec', 'SRGNN']):
    """运行所有模型和数据集组合"""
    results = []
    
    for dataset in datasets:
        for model in models:
            try:
                result = run_model(model, dataset)
                results.append(result)
            except Exception as e:
                print(set_color(f'\n错误: {model} 在 {dataset} 运行失败: {str(e)}', 'red'))
                continue
    
    # 打印汇总结果
    print(set_color('\n' + '='*80, 'green'))
    print(set_color('所有实验结果汇总', 'green'))
    print(set_color('='*80, 'green'))
    
    for result in results:
        print(set_color(f"\n{result['model']} - {result['dataset']}", 'yellow'))
        print(f"测试集结果: {result['test_result']}")
    
    return results


if __name__ == '__main__':
    # 模型配置（可根据需要调整）
    model_configs = {
        'SASRec': {
            'n_layers': 2,
            'n_heads': 2,
            'hidden_size': 64,
            'inner_size': 256,
            'hidden_dropout_prob': 0.2,
            'attention_probs_dropout_prob': 0.2,
        },
        'GRU4Rec': {
            'embedding_size': 64,
            'hidden_size': 128,
            'num_layers': 1,
            'dropout_prob': 0.3,
        },
        'SRGNN': {
            'embedding_size': 64,
            'hidden_size': 100,
            'dropout_prob': 0.3,
        }
    }
    
    # 运行所有模型
    all_results = run_all_models(
        datasets=['mooper', 'assist09'],
        models=['SASRec', 'GRU4Rec', 'SRGNN']
    )
