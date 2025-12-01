"""
模型超参数优化模块
支持：网格搜索、随机搜索、贝叶斯优化（Optuna）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class ModelHyperparameterOptimizer:
    """模型超参数优化器"""
    
    def __init__(self, dataset_train, dataset_val, metric='ic', output_suffix=""):
        """
        dataset_train: 训练数据集
        dataset_val: 验证数据集（用于评估）
        metric: 优化目标 ('ic', 'sharpe', 'return')
        output_suffix: 输出文件后缀
        """
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.metric = metric
        self.output_suffix = output_suffix
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """评估参数组合，返回评分"""
        try:
            # 创建模型配置
            model_config = {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": params
            }
            
            # 训练模型
            model = init_instance_by_config(model_config)
            model.fit(self.dataset_train)
            
            # 在验证集上预测
            pred = model.predict(self.dataset_val)
            
            # 获取真实标签
            from qlib.data.dataset.handler import DataHandlerLP
            val_data = self.dataset_val.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_I)
            if isinstance(val_data, dict):
                y_true = val_data.get("label", None)
            else:
                y_true = val_data
            
            if y_true is None:
                return -np.inf
            
            # 处理预测结果
            if isinstance(pred, pd.DataFrame):
                pred_series = pred.iloc[:, 0]
            else:
                pred_series = pd.Series(pred)
            
            # 处理标签
            if isinstance(y_true, pd.DataFrame):
                label_col = [c for c in y_true.columns if str(c).upper().startswith('LABEL')]
                if label_col:
                    y_true_series = y_true[label_col[0]]
                else:
                    y_true_series = y_true.iloc[:, 0]
            else:
                y_true_series = y_true
            
            # 对齐索引
            common_idx = pred_series.index.intersection(y_true_series.index)
            if len(common_idx) < 10:
                return -np.inf
            
            pred_aligned = pred_series.loc[common_idx]
            y_aligned = y_true_series.loc[common_idx]
            
            # 去除NaN
            valid_mask = pred_aligned.notna() & y_aligned.notna() & np.isfinite(pred_aligned) & np.isfinite(y_aligned)
            if valid_mask.sum() < 10:
                return -np.inf
            
            pred_clean = pred_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            # 根据metric计算评分
            if self.metric == 'ic':
                # 计算IC（Pearson相关系数）
                try:
                    ic, _ = pearsonr(pred_clean.values, y_clean.values)
                    if np.isfinite(ic):
                        return abs(ic)  # 使用绝对值，因为IC的正负取决于因子方向
                    else:
                        return -np.inf
                except:
                    return -np.inf
            elif self.metric == 'sharpe':
                # 计算夏普比率（简化版：收益/波动率）
                returns = y_clean.values
                if len(returns) < 2:
                    return -np.inf
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                if std_ret < 1e-10:
                    return -np.inf
                sharpe = mean_ret / std_ret * np.sqrt(252)  # 年化
                return sharpe
            elif self.metric == 'return':
                # 计算平均收益
                returns = y_clean.values
                if len(returns) < 1:
                    return -np.inf
                return np.mean(returns)
            else:
                return -np.inf
                
        except Exception as e:
            print(f"   ⚠️ 参数评估失败: {e}")
            return -np.inf
    
    def grid_search(self, param_grid: Dict[str, List], cv_folds: int = 3) -> Dict[str, Any]:
        """网格搜索"""
        print("\n" + "="*70)
        print("🔍 开始网格搜索超参数优化")
        print("="*70)
        
        from itertools import product
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"   参数空间大小: {len(param_combinations)} 个组合")
        print(f"   评估指标: {self.metric}")
        
        best_score = -np.inf
        best_params = None
        
        for idx, combo in enumerate(param_combinations):
            params = dict(zip(param_names, combo))
            print(f"\n   组合 {idx+1}/{len(param_combinations)}: {params}")
            
            score = self._evaluate_params(params)
            print(f"   评分: {score:.6f}")
            
            self.optimization_history.append({
                'params': params.copy(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"   ✅ 新的最佳参数！评分: {best_score:.6f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        print("\n" + "="*70)
        print("✅ 网格搜索完成")
        print(f"   最佳评分: {best_score:.6f}")
        print(f"   最佳参数: {best_params}")
        print("="*70)
        
        return best_params
    
    def random_search(self, param_distributions: Dict[str, List], n_iter: int = 50) -> Dict[str, Any]:
        """随机搜索"""
        print("\n" + "="*70)
        print("🔍 开始随机搜索超参数优化")
        print("="*70)
        print(f"   搜索次数: {n_iter}")
        print(f"   评估指标: {self.metric}")
        
        import random
        
        best_score = -np.inf
        best_params = None
        
        for idx in range(n_iter):
            # 随机选择参数
            params = {}
            for param_name, param_values in param_distributions.items():
                params[param_name] = random.choice(param_values)
            
            print(f"\n   迭代 {idx+1}/{n_iter}: {params}")
            
            score = self._evaluate_params(params)
            print(f"   评分: {score:.6f}")
            
            self.optimization_history.append({
                'params': params.copy(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"   ✅ 新的最佳参数！评分: {best_score:.6f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        print("\n" + "="*70)
        print("✅ 随机搜索完成")
        print(f"   最佳评分: {best_score:.6f}")
        print(f"   最佳参数: {best_params}")
        print("="*70)
        
        return best_params
    
    def bayesian_optimization(self, n_trials: int = 100) -> Dict[str, Any]:
        """贝叶斯优化（使用Optuna）"""
        try:
            import optuna
        except ImportError:
            print("❌ 未安装 Optuna，请运行: pip install optuna")
            print("   改用随机搜索...")
            return self.random_search(
                param_distributions={
                    'learning_rate': [0.05, 0.1, 0.15],
                    'n_estimators': [200, 300, 500],
                    'num_leaves': [31, 63, 127],
                    'min_child_samples': [10, 20, 30],
                    'reg_alpha': [0.0, 0.1, 0.3],
                    'reg_lambda': [0.0, 0.1, 0.3],
                },
                n_iter=min(n_trials, 50)
            )
        
        print("\n" + "="*70)
        print("🔍 开始贝叶斯优化超参数优化")
        print("="*70)
        print(f"   优化轮数: {n_trials}")
        print(f"   评估指标: {self.metric}")
        
        def objective(trial):
            # 定义参数搜索空间
            params = {
                'loss': 'mse',
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.3),
            }
            
            score = self._evaluate_params(params)
            
            self.optimization_history.append({
                'params': params.copy(),
                'score': score
            })
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params.copy()
        self.best_params['loss'] = 'mse'  # 固定loss
        self.best_score = study.best_value
        
        print("\n" + "="*70)
        print("✅ 贝叶斯优化完成")
        print(f"   最佳评分: {self.best_score:.6f}")
        print(f"   最佳参数: {self.best_params}")
        print("="*70)
        
        return self.best_params

