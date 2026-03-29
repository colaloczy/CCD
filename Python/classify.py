from sklearn.ensemble import VotingClassifier
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
import warnings
warnings.filterwarnings('ignore')
def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    feature_columns = ["Jaccard","Levenshtein", "LDA","LenRatio"]
    
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"Dataset loaded: {len(X)} samples")
    return X, y

def ensemble_models(X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    model_metrics = {
        'RandomForestClassifier': {
            'F1s': [], 'Precisions': [], 'Recalls': [],
            'train_times': [], 'predict_times': []
        },
        'XGBClassifier': {
            'F1s': [], 'Precisions': [], 'Recalls': [],
            'train_times': [], 'predict_times': []
        },
        'BaggingClassifier': {
            'F1s': [], 'Precisions': [], 'Recalls': [],
            'train_times': [], 'predict_times': []
        },
        'StackingClassifier': {
            'F1s': [], 'Precisions': [], 'Recalls': [],
            'train_times': [], 'predict_times': []
        },
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/10")
        train_X, train_Y = X[train_index], y[train_index]
        test_X, test_Y = X[test_index], y[test_index]

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

        for model in [rf_model, xgb_model, bagging_model, stacking_model]:
            model_name = model.__class__.__name__
            
            start_train = time.time()
            model.fit(train_X, train_Y)
            train_time = time.time() - start_train
            
            start_predict = time.time()
            y_pred = model.predict(test_X)
            predict_time = time.time() - start_predict
            
            f1 = f1_score(test_Y, y_pred)
            precision = precision_score(test_Y, y_pred)
            recall = recall_score(test_Y, y_pred)
            
            model_metrics[model_name]['F1s'].append(f1)
            model_metrics[model_name]['Precisions'].append(precision)
            model_metrics[model_name]['Recalls'].append(recall)
            model_metrics[model_name]['train_times'].append(train_time)
            model_metrics[model_name]['predict_times'].append(predict_time)
            
            print(f"{model_name}:")
            print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Train time: {train_time:.2f}s, Predict time: {predict_time:.2f}s")

    print("\nAverage Results for Each Model:")
    for model_name in model_metrics:
        f1_mean = np.mean(model_metrics[model_name]['F1s'])
        f1_std = np.std(model_metrics[model_name]['F1s'])
        precision_mean = np.mean(model_metrics[model_name]['Precisions'])
        precision_std = np.std(model_metrics[model_name]['Precisions'])
        recall_mean = np.mean(model_metrics[model_name]['Recalls'])
        recall_std = np.std(model_metrics[model_name]['Recalls'])
        train_time_mean = np.mean(model_metrics[model_name]['train_times'])
        predict_time_mean = np.mean(model_metrics[model_name]['predict_times'])
        
        print(f"\n{model_name}:")
        print(f"  F1: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"  Recall: {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"  Avg Train time: {train_time_mean:.2f}s")
        print(f"  Avg Predict time: {predict_time_mean:.2f}s")

def main():
    input_file = 'features.csv'
    
    print("Starting code clone detection...")
    start = time.time()
    
    X, y = load_data(input_file)
    ensemble_models(X, y)
    
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()
