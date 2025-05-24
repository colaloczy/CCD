import csv
from itertools import islice
import numpy as np
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


def feature_extraction_all(feature_csv, feature_indexes=None):
    """Extract features from CSV.
    
    Args:
        feature_csv: Path to the CSV file containing features
        feature_indexes: List of feature indexes to extract. If None, extract all features.
                        Feature indexes are 0-based relative to the feature columns
                        (i.e., starting from column 2 in the CSV)
    
    Returns:
        List of feature vectors
    """
    features = []
    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        # Optional: Read header row to get feature names
        # header = next(data)
        # feature_names = header[2:]  # Skip the first two columns
        next(data)  # Skip header row
        
        for line in islice(data, 0, None):
            try:
                if feature_indexes is None:
                    # Extract all features (original behavior)
                    feature_last = [float(i) for i in line[2:]]
                else:
                    # Extract only the specified features
                    all_features = [float(i) for i in line[2:]]
                    feature_last = [all_features[idx] for idx in feature_indexes]
                features.append(feature_last)
            except Exception as e:
                # print(f"Error processing line: {e}")
                pass
    print('len of features:', len(features))
    print(features[:5])
    return features

def obtain_dataset(dir_path):
    print("--------------------------------------all--------------------------------------")
    nonclone_featureCSV = dir_path + 'nonclone_token_JaccardJaroLeven-RaLDA.csv'
    clone_featureCSV = dir_path + 'clone_token_JaccardJaroLeven-RaLDA.csv'

    Vectors = []
    Labels = []

    feacture_indexs = [0,1,2,3]
    """
    0     -   t1_sim Jaccard
    1     -   t3_sim Jaro
    2     -   t6_sim Leven-ratio
    3     -   t9_sim Lda
    """
    nonclone_features = feature_extraction_all(nonclone_featureCSV, feacture_indexs)
    clone_features = feature_extraction_all(clone_featureCSV, feacture_indexs)

    Vectors.extend(nonclone_features)
    Labels.extend([0 for _ in range(len(nonclone_features))])

    Vectors.extend(clone_features)
    Labels.extend([1 for _ in range(len(clone_features))])

    print('len of Vectors:', len(Vectors))
    print('len of Labels:', len(Labels))

    return Vectors, Labels

def random_features(vectors, labels):
    Vec_Lab = []
    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)
    random.shuffle(Vec_Lab)
    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

def ensemble_models(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    
    # 初始化5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_metrics = {
        'RandomForestClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'XGBClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'BaggingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
        'StackingClassifier': {'F1s': [], 'Precisions': [], 'Recalls': [], 'Times': []},
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/5")
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        # 每个fold重新初始化模型
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

        # 训练并评估每个模型
        for model in [rf_model, xgb_model, bagging_model, stacking_model]:
            start_time = time.time()  # 新增计时开始
            model.fit(train_X, train_Y)
            elapsed = time.time() - start_time  # 计算耗时
            
            y_pred = model.predict(test_X)
            f1 = f1_score(test_Y, y_pred)
            precision = precision_score(test_Y, y_pred)
            recall = recall_score(test_Y, y_pred)
            
            model_name = model.__class__.__name__
            model_metrics[model_name]['F1s'].append(f1)
            model_metrics[model_name]['Precisions'].append(precision)
            model_metrics[model_name]['Recalls'].append(recall)
            model_metrics[model_name]['Times'].append(elapsed)  # 记录时间
            
            print(f"{model_name} - Time: {elapsed:.2f}s | F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 输出平均结果（添加时间统计）
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
        print(f"Training Time: {time_mean:.2f}s ± {time_std:.2f}s")  # 新增时间输出
        
def main1():
    print("-------------------------------------jaccardjaroleven-ratiolda-------------------------------------")
    print("--------------------------------------main--------------------------------------")
    dir_path = '/Java/'
    Vectors, Labels = obtain_dataset(dir_path)
    vectors, labels = random_features(Vectors, Labels)

    start = time.time()
    ensemble_models(vectors, labels)
    end = time.time()
    print("\nTotal time:", end - start)

if __name__ == '__main__':
    main1()