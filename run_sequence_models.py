"""
序列推荐模型运行脚本 - 使用 RecBole
支持单独运行每个模型并立即保存结果
"""
import os
import sys
import pickle
import json
from datetime import datetime

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, get_model, get_trainer, set_color
    from recbole.data.interaction import Interaction
except ImportError as e:
    print(f"RecBole 导入失败: {e}")
    print("请先安装 RecBole: pip install recbole")
    sys.exit(1)


def run_single_model(model_name, dataset_name='assist09', save_dir='./results'):
    """
    运行单个模型并立即保存结果
    
    Args:
        model_name: 模型名称 (SASRec, GRU4Rec, SRGNN)
        dataset_name: 数据集名称
        save_dir: 结果保存目录
    """
    print(f'\n{"="*80}')
    print(f'运行模型: {model_name} | 数据集: {dataset_name}')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*80}\n')
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('./saved_models', exist_ok=True)
    
    # 配置参数
    parameter_dict = {
        'data_path': './dataset/',
        'dataset': dataset_name,
        'data_field_separator': '\t',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp', 'correctness']
        },
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'max_item_list_len': 50,
        'train_neg_sample_args': None,  # CE loss 不需要负采样
        
        # 评估设置 - 序列推荐需要使用 leave-one-out 评估
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_size': 1,
            'leave_one_num': 2,
            'mode': 'full'
        },
        'metrics': ['Hit', 'NDCG', 'MRR'],
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
            'best_valid_score': best_valid_score,
            'best_valid_result': best_valid_result,
            'test_result': test_result,
        }
        
        # 打印结果
        print(f'\n{"="*80}')
        print(f'{model_name} 测试结果:')
        print(f'{"="*80}')
        for metric, value in test_result.items():
            print(f'  {metric}: {value:.4f}')
        
        # 保存结果到 JSON
        result_file = os.path.join(save_dir, f'{dataset_name}_{model_name.lower()}_result.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            # 转换 numpy 类型为 Python 原生类型
            def convert_to_serializable(obj):
                import numpy as np
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                return obj
            
            json.dump(convert_to_serializable(result), f, indent=2, ensure_ascii=False)
        print(f'\n结果已保存到: {result_file}')
        
        # 保存结果到 PKL
        pkl_file = os.path.join(save_dir, f'{dataset_name}_{model_name.lower()}_result.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(result, f)
        print(f'结果已保存到: {pkl_file}')
        
        # 保存模型权重
        model_file = f'./saved_models/{dataset_name}_{model_name.lower()}_model.pth'
        torch_save = False
        try:
            import torch
            torch.save(model.state_dict(), model_file)
            torch_save = True
            print(f'模型权重已保存到: {model_file}')
        except Exception as e:
            print(f'注意: 无法保存 PyTorch 权重 ({e})')
            # 尝试保存整个模型对象
            try:
                with open(model_file.replace('.pth', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                print(f'模型对象已保存到: {model_file.replace(".pth", ".pkl")}')
            except Exception as e2:
                print(f'警告: 无法保存模型 ({e2})')
        
        return result
        
    except Exception as e:
        print(f'\n错误: {model_name} 运行失败')
        print(f'错误信息: {str(e)}')
        import traceback
        traceback.print_exc()
        return None


def print_usage():
    """打印使用说明"""
    print("""
使用说明:
---------
运行单个模型:
  python run_sequence_models.py --model SASRec --dataset mooper
  python run_sequence_models.py --model GRU4Rec --dataset mooper
  python run_sequence_models.py --model SRGNN --dataset mooper

支持的模型:
  - SASRec: Self-Attentive Sequential Recommendation
  - GRU4Rec: Session-based Recommendations with RNNs
  - SRGNN: Session-based Recommendation with Graph Neural Networks

参数说明:
  --model    : 模型名称 (必需)
  --dataset  : 数据集名称 (默认: mooper)
  --save_dir : 结果保存目录 (默认: ./results)
  --all      : 运行所有模型

示例:
  # 运行 SASRec
  python run_sequence_models.py --model SASRec

  # 运行所有模型
  python run_sequence_models.py --all
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行序列推荐模型')
    parser.add_argument('--model', type=str, default=None, 
                       help='模型名称: SASRec, GRU4Rec, SRGNN')
    parser.add_argument('--dataset', type=str, default='mooper',
                       help='数据集名称')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 默认运行所有模型
    if args.model:
        # 运行单个模型
        run_single_model(args.model, args.dataset, args.save_dir)
    else:
        # 运行所有模型（默认行为）
        print(f'\n{"#"*80}')
        print('# 开始运行所有序列推荐模型')
        print(f'# 数据集: {args.dataset}')
        print(f'# 开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'{"#"*80}\n')
        
        models = ['SASRec', 'GRU4Rec', 'SRGNN']
        all_results = {}
        
        for i, model in enumerate(models, 1):
            print(f'\n\n{"*"*80}')
            print(f'* 进度: [{i}/{len(models)}] 正在运行 {model}')
            print(f'{"*"*80}\n')
            
            result = run_single_model(model, args.dataset, args.save_dir)
            if result:
                all_results[model] = result
                print(f'\n✓ {model} 完成！结果已保存。')
            else:
                print(f'\n✗ {model} 运行失败！')
        
        # 保存汇总结果
        summary_file = os.path.join(args.save_dir, f'{args.dataset}_all_sequence_results.pkl')
        with open(summary_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        # 打印最终汇总
        print(f'\n\n{"="*80}')
        print('所有模型运行完成！结果汇总:')
        print(f'{"="*80}')
        print(f'数据集: {args.dataset}\n')
        
        for model, result in all_results.items():
            print(f'\n【{model}】')
            if 'test_result' in result:
                for metric, value in result['test_result'].items():
                    print(f'  {metric}: {value:.4f}')
        
        print(f'\n\n保存文件:')
        print(f'  - 汇总结果: {summary_file}')
        print(f'  - 单独结果: {args.save_dir}/{args.dataset}_*_result.json')
        print(f'  - 模型权重: ./saved_models/{args.dataset}_*_model.pth')
        print(f'\n完成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'{"="*80}\n')
