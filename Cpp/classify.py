from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    feature_columns = [
        "Jaccard", "Levenshtein",
        "LDA", "LenRatio" ]
    X = df[feature_columns].values
    y = df['Label'].values
    
    print(f"Dataset loaded: {len(X)} samples")
    return X, y

def ensemble_models(X, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    models = {
        'RandomForest': {
            'f1': [], 'precision': [], 'recall': [],
            'train_time': [], 'predict_time': [], 'total_time': []
        },
        'XGBoost': {
            'f1': [], 'precision': [], 'recall': [],
            'train_time': [], 'predict_time': [], 'total_time': []
        },
        'Bagging': {
            'f1': [], 'precision': [], 'recall': [],
            'train_time': [], 'predict_time': [], 'total_time': []
        },
        'Stacking': {
            'f1': [], 'precision': [], 'recall': [],
            'train_time': [], 'predict_time': [], 'total_time': []
        }
    }
    
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/10")
        print('='*50)
        
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        
        print("\n1. Training RandomForest...")
        rf_model = RandomForestClassifier(max_depth=32, random_state=42)
        
        start_train = time.time()
        rf_model.fit(train_X, train_Y)
        rf_train_time = time.time() - start_train
        
        start_predict = time.time()
        y_pred = rf_model.predict(test_X)
        rf_predict_time = time.time() - start_predict
        
        rf_total_time = rf_train_time + rf_predict_time
        
        f1 = f1_score(test_Y, y_pred)
        precision = precision_score(test_Y, y_pred)
        recall = recall_score(test_Y, y_pred)
        
        models['RandomForest']['f1'].append(f1)
        models['RandomForest']['precision'].append(precision)
        models['RandomForest']['recall'].append(recall)
        models['RandomForest']['train_time'].append(rf_train_time)
        models['RandomForest']['predict_time'].append(rf_predict_time)
        models['RandomForest']['total_time'].append(rf_total_time)
        
        print(f"RandomForest - F1: {f1:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, Train Time: {rf_train_time:.4f}s, "
              f"Predict Time: {rf_predict_time:.4f}s, Total Time: {rf_total_time:.4f}s")
        
        print("\n2. Training XGBoost...")
        xgb_model = XGBClassifier(
            learning_rate=0.2, 
            max_depth=32, 
            n_estimators=200, 
            random_state=42,
            verbosity=0
        )
        
        start_train = time.time()
        xgb_model.fit(train_X, train_Y)
        xgb_train_time = time.time() - start_train
        
        start_predict = time.time()
        y_pred = xgb_model.predict(test_X)
        xgb_predict_time = time.time() - start_predict
        
        xgb_total_time = xgb_train_time + xgb_predict_time
        
        f1 = f1_score(test_Y, y_pred)
        precision = precision_score(test_Y, y_pred)
        recall = recall_score(test_Y, y_pred)
        
        models['XGBoost']['f1'].append(f1)
        models['XGBoost']['precision'].append(precision)
        models['XGBoost']['recall'].append(recall)
        models['XGBoost']['train_time'].append(xgb_train_time)
        models['XGBoost']['predict_time'].append(xgb_predict_time)
        models['XGBoost']['total_time'].append(xgb_total_time)
        
        print(f"XGBoost - F1: {f1:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, Train Time: {xgb_train_time:.4f}s, "
              f"Predict Time: {xgb_predict_time:.4f}s, Total Time: {xgb_total_time:.4f}s")
        
        print("\n3. Training Bagging...")
        bagging_model = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=32, random_state=42),
            max_features=0.5,
            max_samples=1.0,
            n_estimators=50,
            random_state=42
        )
        
        start_train = time.time()
        bagging_model.fit(train_X, train_Y)
        bagging_train_time = time.time() - start_train
        
        start_predict = time.time()
        y_pred = bagging_model.predict(test_X)
        bagging_predict_time = time.time() - start_predict
        
        bagging_total_time = bagging_train_time + bagging_predict_time
        
        f1 = f1_score(test_Y, y_pred)
        precision = precision_score(test_Y, y_pred)
        recall = recall_score(test_Y, y_pred)
        
        models['Bagging']['f1'].append(f1)
        models['Bagging']['precision'].append(precision)
        models['Bagging']['recall'].append(recall)
        models['Bagging']['train_time'].append(bagging_train_time)
        models['Bagging']['predict_time'].append(bagging_predict_time)
        models['Bagging']['total_time'].append(bagging_total_time)
        
        print(f"Bagging - F1: {f1:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, Train Time: {bagging_train_time:.4f}s, "
              f"Predict Time: {bagging_predict_time:.4f}s, Total Time: {bagging_total_time:.4f}s")
        
        print("\n4. Training Stacking...")
        stacking_model = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(max_depth=32, random_state=42)),
                ('xgb', XGBClassifier(
                    learning_rate=0.2, 
                    max_depth=32, 
                    n_estimators=200, 
                    random_state=42,
                    verbosity=0
                )),
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
        
        start_train = time.time()
        stacking_model.fit(train_X, train_Y)
        stacking_train_time = time.time() - start_train
        
        start_predict = time.time()
        y_pred = stacking_model.predict(test_X)
        stacking_predict_time = time.time() - start_predict
        
        stacking_total_time = stacking_train_time + stacking_predict_time
        
        f1 = f1_score(test_Y, y_pred)
        precision = precision_score(test_Y, y_pred)
        recall = recall_score(test_Y, y_pred)
        
        models['Stacking']['f1'].append(f1)
        models['Stacking']['precision'].append(precision)
        models['Stacking']['recall'].append(recall)
        models['Stacking']['train_time'].append(stacking_train_time)
        models['Stacking']['predict_time'].append(stacking_predict_time)
        models['Stacking']['total_time'].append(stacking_total_time)
        
        print(f"Stacking - F1: {f1:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, Train Time: {stacking_train_time:.4f}s, "
              f"Predict Time: {stacking_predict_time:.4f}s, Total Time: {stacking_total_time:.4f}s")
        
        fold_total_time = (rf_total_time + xgb_total_time + 
                         bagging_total_time + stacking_total_time)
        print(f"\nFold {fold} Total Time: {fold_total_time:.4f}s")
    
    print("\n" + "="*60)
    print("AVERAGE PERFORMANCE (10-Fold Cross Validation)")
    print("="*60)
    
    for model_name in models:
        f1_mean = np.mean(models[model_name]['f1'])
        f1_std = np.std(models[model_name]['f1'])
        precision_mean = np.mean(models[model_name]['precision'])
        precision_std = np.std(models[model_name]['precision'])
        recall_mean = np.mean(models[model_name]['recall'])
        recall_std = np.std(models[model_name]['recall'])
        train_time_mean = np.mean(models[model_name]['train_time'])
        train_time_std = np.std(models[model_name]['train_time'])
        predict_time_mean = np.mean(models[model_name]['predict_time'])
        predict_time_std = np.std(models[model_name]['predict_time'])
        total_time_mean = np.mean(models[model_name]['total_time'])
        total_time_std = np.std(models[model_name]['total_time'])
        
        print(f"\n{model_name}:")
        print(f"  F1:        {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"  Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"  Train Time: {train_time_mean:.4f}s ± {train_time_std:.4f}")
        print(f"  Predict Time: {predict_time_mean:.4f}s ± {predict_time_std:.4f}")
        print(f"  Total Time: {total_time_mean:.4f}s ± {total_time_std:.4f}")

def main():
    input_file = 'cpp-codenet/features.csv'
    print("Starting code clone detection...")
    total_start = time.time()
    
    X, y = load_data(input_file)
    ensemble_models(X, y)
    
    print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")

if __name__ == '__main__':
    main()
