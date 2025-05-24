from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
import pandas as pd
from sklearn.feature_selection import RFE

def load_data(file_path):
    """加载特征数据"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    feature_columns = [
        'Jaccard',  'Jaro', 
        'Levenshtein_ratio',
        'lda_sim'
    ]
    
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"Dataset loaded: {len(X)} samples")
    return X, y

def ensemble_models(X, y):
    """训练和评估模型"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 初始化指标存储结构
    model_names = ['Random Forest', 'XGBoost', 'Bagging', 'Stacking']
    metrics = {name: {'F1s': [], 'Precisions': [], 'Recalls': []} for name in model_names}

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\n{'='*30}\nFold {fold}/5\n{'='*30}")
        train_X, train_Y = X[train_index], y[train_index]
        test_X, test_Y = X[test_index], y[test_index]

        # 定义模型列表（每个fold重新初始化）
        models = [
            ('Random Forest', RandomForestClassifier(max_depth=32, random_state=42)),
            ('XGBoost', XGBClassifier(learning_rate=0.2, max_depth=32, n_estimators=200, use_label_encoder=False, eval_metric='logloss')),
            ('Bagging', BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=32), 
                max_features=0.5, 
                max_samples=1.0, 
                n_estimators=50,
                random_state=42
            )),
            ('Stacking', StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(max_depth=32, random_state=42)),
                    ('xgb', XGBClassifier(learning_rate=0.2, max_depth=32, n_estimators=200, use_label_encoder=False, eval_metric='logloss')),
                    ('bagging', BaggingClassifier(
                        base_estimator=DecisionTreeClassifier(max_depth=32), 
                        max_features=0.5, 
                        max_samples=1.0, 
                        n_estimators=50,
                        random_state=42
                    ))
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3
            ))
        ]

        # 训练和评估每个模型
        for model_name, model in models:
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            # 训练模型
            model.fit(train_X, train_Y)
            
            # 预测和评估
            y_pred = model.predict(test_X)
            train_time = time.time() - start_time
            
            # 记录指标
            metrics[model_name]['F1s'].append(f1_score(test_Y, y_pred))
            metrics[model_name]['Precisions'].append(precision_score(test_Y, y_pred))
            metrics[model_name]['Recalls'].append(recall_score(test_Y, y_pred))
            
            # 打印当前fold结果
            print(f"{model_name} - Fold {fold}:")
            print(f"  F1: {metrics[model_name]['F1s'][-1]:.4f}")
            print(f"  Precision: {metrics[model_name]['Precisions'][-1]:.4f}")
            print(f"  Recall: {metrics[model_name]['Recalls'][-1]:.4f}")
            print(f"  Training Time: {train_time:.2f}s")

    # 打印最终结果
    print("\n\n"+'='*30)
    print("Final Evaluation Results")
    print('='*30)
    for model_name in model_names:
        avg_f1 = np.mean(metrics[model_name]['F1s'])
        std_f1 = np.std(metrics[model_name]['F1s'])
        avg_precision = np.mean(metrics[model_name]['Precisions'])
        std_precision = np.std(metrics[model_name]['Precisions'])
        avg_recall = np.mean(metrics[model_name]['Recalls'])
        std_recall = np.std(metrics[model_name]['Recalls'])
        
        print(f"\n{model_name}:")
        print(f"  F1:       {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f} ± {std_recall:.4f}")

def main():
    input_file = 'cs_token.csv'
    
    print("Starting code clone detection...")
    start = time.time()
    
    # 加载数据
    X, y = load_data(input_file)
    
    # 训练和评估模型
    ensemble_models(X, y)
    
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()
