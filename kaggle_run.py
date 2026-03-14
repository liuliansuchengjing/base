"""
Kaggle 运行脚本 - 序列推荐模型
使用方法：
1. 上传此文件和数据集到 Kaggle
2. 在 Kaggle Notebook 中运行此脚本
"""

# ============== Kaggle 环境设置 ==============
import os
import sys

# Kaggle 环境检测
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    # Kaggle 路径设置
    WORK_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input'  # 需要上传数据到 Kaggle Dataset
    OUTPUT_DIR = '/kaggle/working/results'
    
    print("检测到 Kaggle 环境")
    print(f"工作目录: {WORK_DIR}")
    print(f"数据目录: {DATA_DIR}")
    
    # 安装依赖（Kaggle 默认已有大部分依赖）
    print("\n安装 RecBole...")
    os.system('pip install recbole -q')
else:
    # 本地环境
    WORK_DIR = '.'
    DATA_DIR = './dataset'
    OUTPUT_DIR = './results'

# ============== 导入依赖 ==============
import pickle
import json
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer
import torch


# ============== 配置 ==============
DATASET_NAME = 'mooper'  # 数据集名称
MODELS = ['SASRec', 'GRU4Rec', 'SRGNN']  # 要运行的模型
SAVE_DIR = os.path.join(WORK_DIR, 'results')
MODEL_DIR = os.path.join(WORK_DIR, 'saved_models')

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============== 运行单个模型 ==============
def run_single_model(model_name, dataset_name, data_path):
    """运行单个模型并保存结果"""
    print(f'\n{"="*80}')
    print(f'运行模型: {model_name} | 数据集: {dataset_name}')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*80}\n')
    
    # 配置参数
    parameter_dict = {
        'data_path': data_path,
        'dataset': dataset_name,
        'data_field_separator': '\t',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp', 'correctness']
        },
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'max_item_list_len': 50,
        'train_neg_sample_args': None,
        
        # 评估设置
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'order': 'TO',
            'group_size': 1,
            'leave_one_num': 1,
            'mode': 'valid'
        },
        'metrics': ['Hit', 'NDCG', 'MRR', 'MAP'],
        'topk': [5, 10, 20],
        'valid_metric': 'Hit@10',
        
        # 训练设置
        'epochs': 50,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 0.001,
        'embedding_size': 64,
        
        # 其他设置
        'seed': 2023,
        'reproducibility': True,
        'state': 'INFO',
        'show_progress': True,
    }
    
    # 模型特定参数
    if model_name == 'SASRec':
        parameter_dict.update({
            'n_heads': 2,
            'n_layers': 2,
            'hidden_size': 64,
            'dropout_prob': 0.2,
        })
    elif model_name == 'GRU4Rec':
        parameter_dict.update({
            'hidden_size': 64,
            'num_layers': 1,
            'dropout_prob': 0.2,
        })
    elif model_name == 'SRGNN':
        parameter_dict.update({
            'hidden_size': 64,
            'num_layers': 1,
            'step': 1,
        })
    
    try:
        # 初始化
        init_seed(parameter_dict['seed'], parameter_dict['reproducibility'])
        
        # 加载配置和数据
        print("加载配置和数据...")
        config = Config(model=model_name, config_dict=parameter_dict)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        print(f"数据集信息:")
        print(f"  用户数: {dataset.user_num}")
        print(f"  物品数: {dataset.item_num}")
        print(f"  交互数: {dataset.inter_num}")
        
        # 初始化模型
        print(f"\n初始化 {model_name} 模型...")
        model = get_model(config['model'])(config, dataset).to(config['device'])
        
        # 训练
        print(f"\n开始训练 {model_name}...")
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=config['show_progress']
        )
        
        # 测试
        print(f"\n开始测试 {model_name}...")
        test_result = trainer.evaluate(test_data)
        
        # 整理结果
        result = {
            'model': model_name,
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_valid_score': float(best_valid_score) if best_valid_score else None,
            'best_valid_result': {k: float(v) for k, v in best_valid_result.items()} if best_valid_result else {},
            'test_result': {k: float(v) for k, v in test_result.items()},
        }
        
        # 打印结果
        print(f'\n{"="*80}')
        print(f'{model_name} 测试结果:')
        print(f'{"="*80}')
        for metric, value in test_result.items():
            print(f'  {metric}: {value:.4f}')
        
        # 保存结果
        result_file = os.path.join(SAVE_DIR, f'{dataset_name}_{model_name.lower()}_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'\n结果已保存到: {result_file}')
        
        # 保存模型权重
        model_file = os.path.join(MODEL_DIR, f'{dataset_name}_{model_name.lower()}_model.pth')
        torch.save(model.state_dict(), model_file)
        print(f'模型权重已保存到: {model_file}')
        
        return result
        
    except Exception as e:
        print(f'\n错误: {model_name} 运行失败')
        print(f'错误信息: {str(e)}')
        import traceback
        traceback.print_exc()
        return None


# ============== 主函数 ==============
def main():
    print(f'\n{"#"*80}')
    print('# Kaggle 序列推荐模型实验')
    print(f'# 数据集: {DATASET_NAME}')
    print(f'# 模型: {", ".join(MODELS)}')
    print(f'# 开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"#"*80}\n')
    
    # 数据路径
    if IS_KAGGLE:
        # Kaggle 中数据集路径需要根据实际上传的数据集名称调整
        # 假设数据集上传后路径为 /kaggle/input/mooper/mooper/
        data_path = '/kaggle/input'
        
        # 检查数据是否存在
        dataset_path = os.path.join(data_path, DATASET_NAME, DATASET_NAME)
        if not os.path.exists(dataset_path):
            print(f"错误: 数据集不存在于 {dataset_path}")
            print("请确保已上传数据集到 Kaggle")
            return
    else:
        data_path = './dataset'
    
    # 运行所有模型
    all_results = {}
    
    for i, model in enumerate(MODELS, 1):
        print(f'\n\n{"*"*80}')
        print(f'* 进度: [{i}/{len(MODELS)}] 正在运行 {model}')
        print(f'{"*"*80}\n')
        
        result = run_single_model(model, DATASET_NAME, data_path)
        if result:
            all_results[model] = result
            print(f'\n✓ {model} 完成！')
        else:
            print(f'\n✗ {model} 失败！')
    
    # 保存汇总结果
    summary_file = os.path.join(SAVE_DIR, f'{DATASET_NAME}_all_results.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # 保存汇总 JSON
    summary_json = os.path.join(SAVE_DIR, f'{DATASET_NAME}_all_results.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印最终汇总
    print(f'\n\n{"="*80}')
    print('实验完成！结果汇总:')
    print(f'{"="*80}')
    
    for model, result in all_results.items():
        print(f'\n【{model}】')
        if 'test_result' in result:
            for metric, value in result['test_result'].items():
                print(f'  {metric}: {value:.4f}')
    
    print(f'\n\n保存位置:')
    print(f'  - 结果: {SAVE_DIR}/')
    print(f'  - 权重: {MODEL_DIR}/')
    print(f'\n完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
