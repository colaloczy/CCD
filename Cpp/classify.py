import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
import pandas as pd

def load_data(file_path):
    """加载特征数据"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    feature_columns = [
    'Jaccard', 'Jaro', 
    'Levenshtein_ratio',
    'LDA_similarity']
    
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"Dataset loaded: {len(X)} samples")
    return X, y

def ensemble_models(X, y):
    """训练和评估模型（5折交叉验证）"""
    # 初始化5折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储各模型指标
    models = {
        'RandomForest': {'f1': [], 'precision': [], 'recall': []},
        'XGBoost': {'f1': [], 'precision': [], 'recall': []},
        'Bagging': {'f1': [], 'precision': [], 'recall': []},
        'Stacking': {'f1': [], 'precision': [], 'recall': []}
    }
    
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        print(f"\nFold {fold}/5")
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        
        # 训练Stacking集成模型
        stacking_model = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(max_depth=32, random_state=42)),
                ('xgb', XGBClassifier(learning_rate=0.2, max_depth=32, 
                                     n_estimators=200, random_state=42)),
                ('bagging', BaggingClassifier(
                    base_estimator=DecisionTreeClassifier(max_depth=32, random_state=42),
                    max_features=0.5,
                    max_samples=1.0,
                    n_estimators=50,
                    random_state=42
                ))
            ],
            final_estimator=LogisticRegression(),
            cv=3
        )
        
        # 训练模型
        start_train = time.time()
        stacking_model.fit(train_X, train_Y)
        train_time = time.time() - start_train
        
        # 从Stacking模型中提取基模型
        rf_model = stacking_model.estimators_[0]
        xgb_model = stacking_model.estimators_[1]
        bagging_model = stacking_model.estimators_[2]
        
        # 评估基模型
        for model_name, model in zip(['RandomForest', 'XGBoost', 'Bagging'],
                                    [rf_model, xgb_model, bagging_model]):
            start_test = time.time()
            y_pred = model.predict(test_X)
            test_time = time.time() - start_test
            
            f1 = f1_score(test_Y, y_pred)
            precision = precision_score(test_Y, y_pred)
            recall = recall_score(test_Y, y_pred)
            
            models[model_name]['f1'].append(f1)
            models[model_name]['precision'].append(precision)
            models[model_name]['recall'].append(recall)
            
            print(f"{model_name} - F1: {f1:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, Test Time: {test_time:.2f}s")
        
        # 评估Stacking模型
        start_test = time.time()
        y_pred = stacking_model.predict(test_X)
        test_time = time.time() - start_test
        
        f1 = f1_score(test_Y, y_pred)
        precision = precision_score(test_Y, y_pred)
        recall = recall_score(test_Y, y_pred)
        
        models['Stacking']['f1'].append(f1)
        models['Stacking']['precision'].append(precision)
        models['Stacking']['recall'].append(recall)
        
        print(f"Stacking - F1: {f1:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, Test Time: {test_time:.2f}s")
    
    # 计算并输出平均结果
    print("\n\nAverage Performance:")
    for model_name in models:
        f1_mean = np.mean(models[model_name]['f1'])
        f1_std = np.std(models[model_name]['f1'])
        precision_mean = np.mean(models[model_name]['precision'])
        precision_std = np.std(models[model_name]['precision'])
        recall_mean = np.mean(models[model_name]['recall'])
        recall_std = np.std(models[model_name]['recall'])
        
        print(f"\n{model_name}:")
        print(f"  F1:      {f1_mean:.4f} (±{f1_std:.4f})")
        print(f"  Precision: {precision_mean:.4f} (±{precision_std:.4f})")
        print(f"  Recall:  {recall_mean:.4f} (±{recall_std:.4f})")

def main():
    input_file = 'poj104_token.csv'
    print("Starting code clone detection...")
    start = time.time()
    
    X, y = load_data(input_file)
    ensemble_models(X, y)
    
    print(f"\nTotal execution time: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    main()