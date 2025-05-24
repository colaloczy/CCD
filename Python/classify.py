import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def load_data(file_path):
    """加载特征数据"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    feature_columns = [
        'Jaccard', 'Jaro',
        'Levenshtein_ratio', 
        'LDA_similarity'
    ]
    
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"Dataset loaded: {len(X)} samples")
    return X, y

def ensemble_models(X, y):
    """训练和评估模型"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_metrics = {
        'RandomForestClassifier': {'F1s': [], 'Precisions': [], 'Recalls': []},
        'XGBClassifier': {'F1s': [], 'Precisions': [], 'Recalls': []},
        'BaggingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': []},
        'StackingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': []},
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/5")
        train_X, train_Y = X[train_index], y[train_index]
        test_X, test_Y = X[test_index], y[test_index]

        # 初始化模型（每个fold使用新实例）
        rf_model = RandomForestClassifier(max_depth=32, random_state=42)
        xgb_model = XGBClassifier(learning_rate=0.2, max_depth=32, n_estimators=200, use_label_encoder=False, eval_metric='logloss')
        bagging_model = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=32), 
            max_features=0.5, 
            max_samples=1.0, 
            n_estimators=50
        )
        stacking_model = StackingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('bagging', bagging_model)
            ],
            final_estimator=LogisticRegression(),
            cv=3
        )

        # 训练并评估每个模型
        for model in [rf_model, xgb_model, bagging_model, stacking_model]:
            model.fit(train_X, train_Y)
            y_pred = model.predict(test_X)
            f1 = f1_score(test_Y, y_pred)
            precision = precision_score(test_Y, y_pred)
            recall = recall_score(test_Y, y_pred)
            
            model_name = model.__class__.__name__
            model_metrics[model_name]['F1s'].append(f1)
            model_metrics[model_name]['Precisions'].append(precision)
            model_metrics[model_name]['Recalls'].append(recall)
            
            print(f"{model_name} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 输出每个模型的平均性能
    print("\nAverage Results for Each Model:")
    for model_name in model_metrics:
        f1_mean = np.mean(model_metrics[model_name]['F1s'])
        f1_std = np.std(model_metrics[model_name]['F1s'])
        precision_mean = np.mean(model_metrics[model_name]['Precisions'])
        precision_std = np.std(model_metrics[model_name]['Precisions'])
        recall_mean = np.mean(model_metrics[model_name]['Recalls'])
        recall_std = np.std(model_metrics[model_name]['Recalls'])
        
        print(f"\n{model_name}:")
        print(f"F1: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")

def main():
    input_file = 'py_features.csv'
    
    print("Starting code clone detection...")
    start = time.time()
    
    X, y = load_data(input_file)
    ensemble_models(X, y)
    
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()