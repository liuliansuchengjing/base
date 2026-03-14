"""
简化版 RecBole 运行脚本 - 仅使用基础模型
"""
import os
import sys

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer


def run_model_simple(model_name, dataset_name):
    """运行单个模型"""
    print(f'\n{"="*80}')
    print(f'运行 {model_name} 在 {dataset_name} 数据集')
    print(f'{"="*80}\n')
    
    # 配置参数
    config_dict = {
        'model': model_name,
        'dataset': dataset_name,
        'data_path': f'./dataset/{dataset_name}',
        
        # 评估设置
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'labeled',
        },
        'metrics': ['Hit', 'NDCG', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'Hit@20',
        
        # 训练参数
        'epochs': 30,
        'train_batch_size': 512,
        'eval_batch_size': 512,
        'learning_rate': 0.001,
        'early_stop': 5,
        
        # 其他设置
        'device': 'cpu',  # 使用 CPU 避免 CUDA 问题
        'seed': 2020,
        'show_progress': True,
        'verbose': True,
    }
    
    # 初始化配置
    config = Config(model=model_name, dataset=dataset_name, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # 创建数据集
    print("创建数据集...")
    dataset = create_dataset(config)
    print(f"数据集信息: {dataset}")
    
    # 数据准备
    print("准备训练/验证/测试数据...")
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 初始化模型
    print(f"初始化模型 {model_name}...")
    model = get_model(config['model'])(config, dataset).to(config['device'])
    print(f"模型结构:\n{model}")
    
    # 训练器
    trainer = get_trainer(config['model_type'])(config, model)
    
    # 训练
    print("开始训练...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    
    # 测试
    print("开始测试...")
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    
    # 输出结果
    print(f'\n{"="*80}')
    print(f'{model_name} 在 {dataset_name} 的结果:')
    print(f'{"="*80}')
    print(f'验证集最佳结果: {best_valid_result}')
    print(f'测试集结果: {test_result}')
    print(f'{"="*80}\n')
    
    return test_result


def main():
    """主函数"""
    datasets = ['mooper', 'assist09']
    models = ['SASRec', 'GRU4Rec', 'SRGNN']
    
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        for model in models:
            try:
                result = run_model_simple(model, dataset)
                all_results[dataset][model] = result
            except Exception as e:
                print(f'\n错误: {model} 在 {dataset} 运行失败')
                print(f'错误信息: {str(e)}')
                import traceback
                traceback.print_exc()
                all_results[dataset][model] = None
    
    # 打印汇总结果
    print('\n' + '='*80)
    print('所有实验结果汇总')
    print('='*80)
    for dataset, models_results in all_results.items():
        print(f'\n数据集: {dataset}')
        for model, result in models_results.items():
            print(f'  {model}: {result}')
    
    return all_results


if __name__ == '__main__':
    main()
