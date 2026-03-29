from sklearn.ensemble import VotingClassifier
import argparse
import csv
from itertools import islice
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')
def parse_options():
    parser = argparse.ArgumentParser(description='Malware Detection.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware feature csv.', required=True, type=str)
    parser.add_argument('-o', '--out', help='The dir_path of output', required=True, type=str)
    args = parser.parse_args()
    return args

def load_dataset(clone_csv, nonclone_csv):
    # Load data with specific column names
    df_clone = pd.read_csv(clone_csv)
    df_non = pd.read_csv(nonclone_csv)
    
    # Ensure we have the correct columns
    feature_columns = ['Jaccard',  'Levenshtein', 'LDA', 'LenRatio']
    df_clone = df_clone[feature_columns + ['Label']]
    df_non = df_non[feature_columns + ['Label']]
    
    df = pd.concat([df_clone, df_non], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def obtain_dataset(dir_path):
    print("----------------------------------------------------------------------------")
    nonclone_featureCSV = dir_path + 'java_nonclone_features_global_lda.csv'
    clone_featureCSV = dir_path + 'java_allclone_features_global_lda.csv'

    # Load and process dataset
    df = load_dataset(clone_featureCSV, nonclone_featureCSV)
    
    # Separate features and labels
    feature_columns = ['Jaccard',  'Levenshtein', 'LDA', 'LenRatio']
    Vectors = df[feature_columns].values.tolist()
    Labels = df['Label'].values.tolist()

    print('\nlen of Vectors:', len(Vectors))
    print('len of Labels:', len(Labels))

    return Vectors, Labels

def ensemble_models(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    
    # Initialize 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    model_metrics = {
        'RandomForestClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'XGBClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'BaggingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'StackingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/10")
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        # Initialize models for each fold
        rf_model = RandomForestClassifier(max_depth=32, random_state=42)
        xgb_model = XGBClassifier(learning_rate=0.2, max_depth=32, n_estimators=200, 
                                 use_label_encoder=False, eval_metric='logloss')
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

        # Train and evaluate each model
        for model in [rf_model, xgb_model, bagging_model, stacking_model]:
            start_time = time.time()
            model.fit(train_X, train_Y)
            elapsed = time.time() - start_time
            
            y_pred = model.predict(test_X)
            f1 = f1_score(test_Y, y_pred)
            precision = precision_score(test_Y, y_pred)
            recall = recall_score(test_Y, y_pred)
            
            model_name = model.__class__.__name__
            model_metrics[model_name]['F1s'].append(f1)
            model_metrics[model_name]['Precisions'].append(precision)
            model_metrics[model_name]['Recalls'].append(recall)
            model_metrics[model_name]['Times'].append(elapsed)
            
            print(f"{model_name} - Time: {elapsed:.2f}s | F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Output average results
    print("\nAverage Results for Each Model:")
    for model_name in model_metrics:
        f1_mean = np.mean(model_metrics[model_name]['F1s'])
        f1_std = np.std(model_metrics[model_name]['F1s'])
        precision_mean = np.mean(model_metrics[model_name]['Precisions'])
        precision_std = np.std(model_metrics[model_name]['Precisions'])
        recall_mean = np.mean(model_metrics[model_name]['Recalls'])
        recall_std = np.std(model_metrics[model_name]['Recalls'])
        time_mean = np.mean(model_metrics[model_name]['Times'])
        time_std = np.std(model_metrics[model_name]['Times'])
        
        print(f"\n{model_name}:")
        print(f"F1: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"Training Time: {time_mean:.2f}s ± {time_std:.2f}s")
        
def main():
    print("--------------------------------------main------------------------------------")
    dir_path = 'java-bcb/'
    vectors, labels = obtain_dataset(dir_path)

    start = time.time()
    ensemble_models(vectors, labels)
    end = time.time()
    print("\nTotal time:", end - start)

if __name__ == '__main__':
    main()
