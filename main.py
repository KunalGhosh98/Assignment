#!/usr/bin/env python3
"""DS Assignment 1 - Generic ML pipeline executor"""
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib
warnings.filterwarnings('ignore')

def is_numeric_dtype(series):
    return pd.api.types.is_numeric_dtype(series)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_preprocessor(df, feature_specs):
    numeric_cols = []
    categorical_cols = []
    for feat in feature_specs:
        name = feat['name']
        if is_numeric_dtype(df[name]):
            numeric_cols.append(name)
        else:
            categorical_cols.append(name)
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    transformers = []
    if numeric_cols:
        transformers.append(('num', Pipeline([('imputer', num_imputer),('scaler', StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', Pipeline([('imputer', cat_imputer),('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols))
    preproc = ColumnTransformer(transformers, remainder='drop')
    return preproc, numeric_cols + categorical_cols

def feature_reduction_step(X_train, y_train, reduction_spec, task, random_state=42):
    method = reduction_spec.get('method', 'none').lower()
    if method == 'none':
        return 'passthrough', None
    elif method == 'corr_with_target':
        k = reduction_spec.get('k', 5)
        if task == 'classification':
            return SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1])), None
        else:
            return SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1])), None
    elif method == 'tree_based':
        n_estimators = reduction_spec.get('n_estimators', 100)
        if task == 'classification':
            base = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        else:
            base = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        return SelectFromModel(base), base
    elif method == 'pca':
        n_comp = reduction_spec.get('n_components', 0.95)
        return PCA(n_components=n_comp), None
    else:
        raise ValueError(f'Unknown reduction method: {method}')

def get_models_from_spec(spec, task):
    models = []
    for algo in spec.get('algorithms', []):
        if not algo.get('is_selected', False):
            continue
        name = algo['name'].lower()
        params = algo.get('hyperparams', {})
        if task == 'classification':
            if 'logistic' in name:
                est = LogisticRegression(max_iter=1000)
                grid = params or {'C': [1.0]}
            elif 'randomforest' in name or 'random_forest' in name:
                est = RandomForestClassifier(random_state=42)
                grid = params or {'n_estimators':[100], 'max_depth':[None]}
            elif 'gradient' in name or 'gb' in name:
                est = GradientBoostingClassifier(random_state=42)
                grid = params or {'n_estimators':[100], 'learning_rate':[0.1]}
            elif 'svm' in name or 'svc' in name:
                est = SVC(probability=True)
                grid = params or {'C':[1.0], 'kernel':['rbf']}
            else:
                print(f"Warning: Unknown classifier '{name}', skipping.")
                continue
        else:
            if 'linear' in name and 'regression' in name:
                est = LinearRegression()
                grid = params or {}
            elif 'randomforest' in name or 'random_forest' in name:
                est = RandomForestRegressor(random_state=42)
                grid = params or {'n_estimators':[100], 'max_depth':[None]}
            elif 'gradient' in name or 'gb' in name:
                est = GradientBoostingRegressor(random_state=42)
                grid = params or {'n_estimators':[100], 'learning_rate':[0.1]}
            else:
                print(f"Warning: Unknown regressor '{name}', skipping.")
                continue
        models.append((algo['name'], est, grid))
    return models

def evaluate(y_true, y_pred, task):
    if task == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    else:
        return {
            'rmse': mean_squared_error(y_true, y_pred, squared=False),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

def main(args):
    data_path = Path(args.data)
    spec_path = Path(args.spec)
    df = pd.read_csv(data_path)
    spec = load_json(spec_path)

    target_col = spec['target']['name']
    task = spec['prediction_type'].lower()
    feature_names = [f['name'] for f in spec['features']]

    X = df[feature_names].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=spec.get('test_size', 0.2), random_state=42, stratify=y if task=='classification' else None)

    preproc, used_features = build_preprocessor(pd.concat([X_train, X_test]), spec['features'])

    reduction_spec = spec.get('feature_reduction', {'method':'none'})
    reducer, reducer_base = feature_reduction_step(X_train, y_train, reduction_spec, task)

    models = get_models_from_spec(spec, task)
    if not models:
        print('No models selected to run. Exiting.')
        return

    for label, estimator, param_grid in models:
        print('='*80)
        print(f'Running model: {label}')
        steps = []
        steps.append(('preproc', preproc))
        if reducer != 'passthrough':
            steps.append(('reduction', reducer))
        steps.append(('estimator', estimator))
        pipeline = Pipeline(steps)

        wrapped_grid = {}
        if param_grid:
            for k, v in param_grid.items():
                if '__' in k:
                    wrapped_grid[k] = v
                else:
                    wrapped_grid[f'estimator__{k}'] = v
        else:
            wrapped_grid = {}

        if wrapped_grid:
            search = GridSearchCV(pipeline, wrapped_grid, cv=spec.get('cv',5), n_jobs=args.n_jobs, scoring=spec.get('scoring', None))
        else:
            search = pipeline

        if isinstance(search, GridSearchCV):
            search.fit(X_train, y_train)
            best = search.best_estimator_
            print('Best params:', search.best_params_)
            pred = best.predict(X_test)
        else:
            search.fit(X_train, y_train)
            pred = search.predict(X_test)

        metrics = evaluate(y_test, pred, task)
        print('Metrics:')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')

        model_fname = f"model_{label.replace(' ', '_')}.joblib"
        try:
            joblib.dump(search if isinstance(search, GridSearchCV) else pipeline, model_fname)
            print(f"Saved model object to {model_fname}")
        except Exception as e:
            print('Could not save model:', e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--spec', required=True, help='Path to JSON spec file')
    parser.add_argument('--n_jobs', type=int, default=1, help='n_jobs for GridSearchCV')
    args = parser.parse_args()
    main(args)
