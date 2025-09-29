from catboost import CatBoostClassifier, Pool
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_rapier_base(y, bk, l, *args, **kwargs):
    return CatBoostClassifier(
        iterations=bk,
        learning_rate=l,
        depth=6,
        loss_function='Logloss',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric='AUC',
        task_type="GPU",
        border_count=254,
        boosting_type="Plain",
        verbose=1000,
        early_stopping_rounds=1000,
        random_seed=42,
        *args,
        **kwargs
    )

def get_rapier_twilight(y, tk, l, *args, **kwargs):
    return CatBoostClassifier(
        iterations=tk,
        learning_rate=l,
        depth=6,
        loss_function='Logloss',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric='AUC',
        task_type="GPU",
        border_count=254,
        boosting_type="Plain",
        verbose=1000,
        early_stopping_rounds=1000,
        random_seed=42,
        *args,
        **kwargs
    )

def get_cb_pred(data, model, cat_features, drop=[]):
    test_pool = Pool(
        data=data.drop(columns=drop),
        cat_features=cat_features
    )
    return model.predict_proba(test_pool)[:, 1]


def generate_drop(sorted_idx, percent):
    drop = []
    for i in range(0, int(len(sorted_idx) * percent)):
        if sorted_idx[i] == 0:
            drop.append('id')
            continue
        drop.append(f'feature_{sorted_idx[i]}')
    return drop


def fitting(path, twilight_factor, bk, bl, tk, tl):
    try:
        current_data = os.listdir(path)
    except Exception:
        return "Не папка"
    else:
        current_data = os.listdir(path)
    # Train data loading
    train_data = [data for data in current_data if data.endswith('train.parquet')][0]
    train_data = pd.read_parquet(path + f'/{train_data}')

    # Test data loading
    test_data = [data for data in current_data if data.endswith('test.parquet')][0]
    test_data = pd.read_parquet(path + f'/{test_data}').drop(columns=['smpl'])
    res = test_data[['id']].copy()

    # BaseModel
    X = train_data.drop(columns=['target', 'smpl'])
    y = train_data['target']
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features
    )
    full_rapier = get_rapier_base(y=y, bk=bk, l=bl)
    full_rapier.fit(train_pool, eval_set=(X_test, y_test), use_best_model=True)
    model_pred = get_cb_pred(test_data, full_rapier, cat_features)
    res['target_0'] = model_pred

    # feature importance of BaseModel
    feature_importance = full_rapier.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    # twilight plunging (core)
    for i, drop_percent in enumerate(twilight_factor):
        drop = generate_drop(sorted_idx, drop_percent)
        X = train_data.drop(columns=['target', 'smpl', *drop])
        y = train_data['target']
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features
        )
        full_rapier = get_rapier_twilight(y=y, tk=tk, l=tl)
        full_rapier.fit(train_pool, eval_set=(X_test, y_test), use_best_model=True)
        model_pred = get_cb_pred(test_data, full_rapier, cat_features, drop)
        res[f'target_{i + 1}'] = model_pred

    # blending
    blended_predictions = res.drop(columns=['id']).mean(axis=1)
    print(blended_predictions)
    res['target'] = blended_predictions
    prediction = res[['id', 'target']].sort_values(by='id', ascending=True)
    return prediction

def model(twilight_factor=[0.5, 0.8], bk=5000, bl=0.05, tk=5000, tl=0.05):
    data = 'data'
    folders = os.listdir(data)
    start_time = time.time()
    for fold in folders:
        data_path = data + f'/{fold}'
        print(f"Model fitting for {fold}...")
        prediction = fitting(data_path, twilight_factor, bk, bl, tk, tl)
        if type(prediction) is not str:
            prediction.to_csv(f"predictions/{fold}.csv", index=False)
            print("Предсказание создано!")
            elapsed_time = time.time() - start_time
            print(f"Время с начала работы: {int(elapsed_time / 60)} минут")
        else:
            print("Невозможно создать предсказание!")
            
if __name__ == "__main__":
    model(twilight_factor=[0.4, 0.5, 0.6, 0.8], bk=15000, bl=0.03, tk=30000, tl=0.02)
