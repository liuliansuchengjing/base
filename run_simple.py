"""
简化版 RecBole 运行脚本 - 使用 run_recbole 快速运行
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from recbole.quick_start import run_recbole


def run_single_model(model_name, dataset_name):
    """运行单个模型"""
    print(f'\n{"="*60}')
    print(f'运行 {model_name} 在 {dataset_name} 数据集')
    print(f'{"="*60}\n')
    
    config_dict = {
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'labeled',
        },
        'metrics': ['Hit', 'NDCG', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'Hit@20',
        'epochs': 50,
        'train_batch_size': 1024,
        'learning_rate': 0.001,
        'early_stop': 5,
    }
    
    run_recbole(
        model=model_name,
        dataset=dataset_name,
        config_dict=config_dict
    )


if __name__ == '__main__':
    datasets = ['mooper', 'assist09']
    models = ['SASRec', 'GRU4Rec', 'SRGNN']
    
    for dataset in datasets:
        for model in models:
            try:
                run_single_model(model, dataset)
            except Exception as e:
                print(f'\n错误: {model} 在 {dataset} 运行失败: {str(e)}')
                import traceback
                traceback.print_exc()
                continue
