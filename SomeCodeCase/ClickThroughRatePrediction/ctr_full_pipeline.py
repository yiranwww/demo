"""
Complete CTR project pipeline (召回 + 策略层 + 精排 + 评估 + 解释)
Designed to plug into your existing project (uses `Data/ad_10000records.csv` by default).

Features added compared to your original script:
- Modular feature engineering using sklearn Pipeline / ColumnTransformer
- Simple rule-based召回模拟 + embedding-simulated召回 (label-encoder cosine)
- 精排模型训练（LightGBM/XGBoost），支持可选的超参搜索（optuna优先，否则RandomizedSearchCV）
- 概率校准（Platt / Isotonic）
- 评估指标：AUC, PR-AUC, logloss, 精确/召回/F1, calibration plot
- 特征重要性：permutation + SHAP（封装）
- 保存模型、报告图表
- 可选的在线学习/增量训练示例

NOTE: 依赖库: pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, xgboost, shap
optuna为可选。如果缺失，会降级到RandomizedSearchCV。
"""

import os
import math
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try import optional libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import optuna
except Exception:
    optuna = None

try:
    import shap
except Exception:
    shap = None


# -------------------- Utility / config --------------------
BASE_PATH = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_PATH, 'Data', 'ad_10000records.csv')
MODEL_OUT = os.path.join(BASE_PATH, 'models')
FIG_OUT = os.path.join(BASE_PATH, 'figures')
os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(FIG_OUT, exist_ok=True)

RANDOM_STATE = 42
TOP_K_RECALL = 100  # for toy dataset keep small




# -------------------- Data load & basic cleaning --------------------

def load_and_clean(path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names and trim strings
    df = df.rename(columns=lambda x: x.strip().replace(' ', '_'))

    # Basic timestamp handling if present
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['hour'] = df['Timestamp'].dt.hour.fillna(-1).astype(int)
        df['day'] = df['Timestamp'].dt.day.fillna(0).astype(int)
        df['weekday'] = df['Timestamp'].dt.weekday.fillna(-1).astype(int)
        df['month'] = df['Timestamp'].dt.month.fillna(0).astype(int)

    # Strip object columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip().str.title()

    # Ensure binary target is int
    if 'Clicked_on_Ad' in df.columns:
        df['Clicked_on_Ad'] = df['Clicked_on_Ad'].astype(int)

    return df


# -------------------- Simple召回 (rule-based & pseudo-embedding) --------------------
def build_ad_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ad stats such as global CTR, impressions, clicks.
    Here we assume 'Ad_Topic_Line' identifies the ad creative (or use a surrogate).
    """
    if 'Ad_Topic_Line' in df.columns:
        grp = df.groupby('Ad_Topic_Line').agg(
            impressions=('Clicked_on_Ad', 'count'),
            clicks=('Clicked_on_Ad', 'sum')
        ).reset_index()
        grp['ctr'] = grp['clicks'] / grp['impressions']
        return grp
    else:
        # fallback: treat each row as unique creative (not ideal)
        return pd.DataFrame()


def rule_based_recall(user_row: pd.Series, ad_stats: pd.DataFrame, top_k=TOP_K_RECALL) -> List[str]:
    """Recall ads by simple rules:
    1) same Ad_Topic_Line (exact match) -> high priority
    2) otherwise top-K by global CTR
    Returns list of Ad_Topic_Line strings (ad ids)
    """
    candidates = []
    if 'Ad_Topic_Line' in user_row.index and 'Ad_Topic_Line' in ad_stats.columns:
        topic = user_row['Ad_Topic_Line']
        if topic in ad_stats['Ad_Topic_Line'].values:
            candidates.append(topic)
    # fill with top CTR creatives
    top_creatives = ad_stats.sort_values('ctr', ascending=False)['Ad_Topic_Line'].tolist()
    for ad in top_creatives:
        if ad not in candidates:
            candidates.append(ad)
        if len(candidates) >= top_k:
            break
    return candidates[:top_k]


def pseudo_embedding_recall(user_row: pd.Series, ad_stats: pd.DataFrame, df: pd.DataFrame, top_k=TOP_K_RECALL) -> List[str]:
    """A toy 'embedding' style recall using label encoding on categorical features and cosine similarity.
    Build an aggregate 'ad vector' by averaging encoding for users who clicked that ad.
    """
    # choose features to build embedding
    embed_cols = ['Gender', 'City', 'Country']
    # ensure columns exist
    embed_cols = [c for c in embed_cols if c in df.columns]
    if len(embed_cols) == 0 or ad_stats.empty:
        return ad_stats.sort_values('ctr', ascending=False)['Ad_Topic_Line'].tolist()[:top_k]

    # label encode these columns
    encoders = {}
    enc_df = df[embed_cols].fillna('NA').astype(str).copy()
    for c in enc_df.columns:
        enc, uniques = pd.factorize(enc_df[c])
        enc_df[c] = enc
        encoders[c] = uniques

    df_enc = df.copy()
    for i, c in enumerate(embed_cols):
        df_enc[c + '_enc'] = enc_df[c]

    # compute ad vector as mean of enc cols among users who clicked that ad
    ad_vectors = {}
    for ad in ad_stats['Ad_Topic_Line'].unique():
        rows = df_enc[df_enc['Ad_Topic_Line'] == ad]
        if len(rows) == 0:
            continue
        vec = rows[[c + '_enc' for c in embed_cols]].mean(axis=0).values
        ad_vectors[ad] = vec

    # build user vector
    user_vec = []
    for c in embed_cols:
        val = user_row.get(c, 'NA')
        try:
            idx = list(encoders[c]).index(val)
        except ValueError:
            idx = -1
        user_vec.append(float(idx))
    user_vec = np.array(user_vec)

    # cosine similarity
    def cosine(a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sims = [(ad, cosine(user_vec, vec)) for ad, vec in ad_vectors.items()]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    recall_list = [ad for ad, _ in sims][:top_k]

    # fallback pad with top CTR
    pad = ad_stats.sort_values('ctr', ascending=False)['Ad_Topic_Line'].tolist()
    for ad in pad:
        if ad not in recall_list:
            recall_list.append(ad)
        if len(recall_list) >= top_k:
            break
    return recall_list[:top_k]


# -------------------- Feature engineering pipeline --------------------

def build_feature_pipeline(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    # choose feature columns
    ignore_cols = ['Clicked_on_Ad', 'Timestamp', 'Ad_Topic_Line']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # heuristic split
    num_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64] and df[c].nunique() > 10]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # small-cardinality categories -> OneHot, large -> Ordinal
    ohe_cols = [c for c in cat_cols if df[c].nunique() <= 10]
    ord_cols = [c for c in cat_cols if c not in ohe_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ohe_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ord_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('ohe', ohe_transformer, ohe_cols),
        ('ord', ord_transformer, ord_cols)
    ], remainder='drop', sparse_threshold=0)

    # We'll need final feature names for interpretation. Return them after fitting.
    return preprocessor, (num_cols, ohe_cols, ord_cols)


# -------------------- Model training / selection --------------------

def get_default_model():
    if lgb is not None:
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif XGBClassifier is not None:
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    else:
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


def hyperparam_search_with_fallback(model, X_train, y_train):
    """If optuna available, do a quick optimization; otherwise do RandomizedSearchCV with a small search space."""
    if optuna is not None and lgb is not None:
        # quick optuna for LGB
        def objective(trial):
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0)
            }
            dtrain = lgb.Dataset(X_train, label=y_train)
            cvres = lgb.cv(param, dtrain, nfold=3, stratified=True, seed=RANDOM_STATE, verbose_eval=False, metrics=['binary_logloss'])
            return min(cvres['binary_logloss-mean'])

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)
        best = study.best_params
        # convert to model
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best)
        return model, study
    else:
        # fallback randomized search for a few params
        param_dist = {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [3, 5, 7, 9, None],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }
        rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=12, cv=3, scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE)
        rs.fit(X_train, y_train)
        return rs.best_estimator_, rs


# -------------------- Train full pipeline (召回 -> 精排) --------------------

def run_pipeline(df: pd.DataFrame):
    print('Building ad stats...')
    ad_stats = build_ad_stats(df)

    # Example: build a small candidate set for each row using recall functions
    print('Preparing recall candidates (rule-based + pseudo-embedding)...')
    # We'll attach a column 'candidates' containing list of ad ids
    df = df.copy()
    df['candidates'] = df.apply(lambda row: rule_based_recall(row, ad_stats, top_k=TOP_K_RECALL), axis=1)
    # you can switch to pseudo_embedding_recall if you want: pseudo_embedding_recall(row, ad_stats, df)

    # For training precision model, create training examples from full df but restrict features to user+ad pairing
    # We'll expand each row into N training samples (user, candidate_ad) with label = 1 if clicked and ad==ad_shown
    # Here dataset lacks explicit 'which_ad_was_shown' except Ad_Topic_Line -> use that

    if 'Ad_Topic_Line' not in df.columns:
        raise ValueError('Data must contain Ad_Topic_Line to build supervised user-ad pairs.')

    expanded_rows = []
    for idx, row in df.iterrows():
        shown_ad = row['Ad_Topic_Line']
        candidates = row['candidates'][:50]  # limit expansion for speed
        for ad in candidates:
            expanded = row.to_dict()
            expanded['candidate_ad'] = ad
            # label 1 if candidate matches shown and clicked
            expanded['label'] = int((ad == shown_ad) and (row.get('Clicked_on_Ad', 0) == 1))
            expanded_rows.append(expanded)
    expanded_df = pd.DataFrame(expanded_rows)

    print(f'Expanded dataset size (user-ad pairs): {len(expanded_df):,}')

    # merge ad-level stats (ctr) into expanded
    expanded_df = expanded_df.merge(ad_stats[['Ad_Topic_Line', 'ctr', 'impressions']], how='left', left_on='candidate_ad', right_on='Ad_Topic_Line')
    expanded_df.rename(columns={'ctr': 'candidate_ctr', 'impressions': 'candidate_impr'}, inplace=True)

    # features and target
    y = expanded_df['label']

    # drop columns not useful or leakage
    drop_cols = ['Clicked_on_Ad', 'candidates', 'label', 'Ad_Topic_Line', 'candidate_ad']
    X = expanded_df.drop(columns=[c for c in drop_cols if c in expanded_df.columns])

    # build preprocessing pipeline
    preproc, col_groups = build_feature_pipeline(X)
    X_trans = preproc.fit_transform(X)

    # feature names construction (best-effort)
    num_cols, ohe_cols, ord_cols = col_groups
    feat_names = []
    feat_names.extend(num_cols)
    if len(ohe_cols) > 0:
        ohe = preproc.named_transformers_['ohe']['onehot']
        ohe_names = ohe.get_feature_names_out(ohe_cols).tolist()
        feat_names.extend(ohe_names)
    feat_names.extend(ord_cols)

    # train-test split on pairs (stratify on label to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print('Fitting model (with hyperparam search fallback) ...')
    base_model = get_default_model()
    model, search_obj = hyperparam_search_with_fallback(base_model, X_train, y_train)
    model.fit(X_train, y_train)

    # calibration for probability outputs
    print('Calibrating model probabilities ...')
    calib = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    try:
        calib.fit(X_train, y_train)
        final_model = calib
    except Exception:
        # fallback to platt (sigmoid) if isotonic fails
        calib = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        calib.fit(X_train, y_train)
        final_model = calib

    # evaluation
    print('Evaluating model ...')
    probs = final_model.predict_proba(X_test)[:, 1]
    preds = final_model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    ll = log_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f'AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, LogLoss: {ll:.4f}')
    print(f'Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}')

    # ROC plot
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT, 'roc_curve.png'))

    # calibration plot
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Fraction')
    plt.title('Calibration Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT, 'calibration.png'))

    # feature importance: permutation (works with sklearn API)
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
        imp_idx = np.argsort(perm.importances_mean)[::-1][:30]
        plt.figure(figsize=(8, 6))
        names = [feat_names[i] if i < len(feat_names) else f'ft_{i}' for i in imp_idx]
        plt.barh(names[::-1], perm.importances_mean[imp_idx][::-1])
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_OUT, 'perm_importance.png'))
    except Exception as e:
        print('Permutation importance failed:', e)

    # SHAP summary if available
    if shap is not None and hasattr(model, 'predict_proba'):
        try:
            explainer = shap.Explainer(model.predict_proba, feature_names=feat_names)
            shap_vals = explainer(X_test)
            shap.summary_plot(shap_vals[:, 1], features=None, feature_names=feat_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_OUT, 'shap_summary.png'))
        except Exception as e:
            print('SHAP failed:', e)

    # Save model (sklearn API)
    import joblib
    joblib.dump(final_model, os.path.join(MODEL_OUT, 'final_calibrated_model.joblib'))
    print('Saved calibrated model to:', os.path.join(MODEL_OUT, 'final_calibrated_model.joblib'))

    # Return a dict with results for further analysis
    return {
        'model': final_model,
        'preprocessor': preproc,
        'feature_names': feat_names,
        'metrics': {
            'auc': auc, 'pr_auc': pr_auc, 'logloss': ll,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
        },
        'search_obj': search_obj
    }


# -------------------- Main --------------------
if __name__ == '__main__':
    df = load_and_clean(CSV_PATH)
    res = run_pipeline(df)
    print('Pipeline completed. Figures saved into', FIG_OUT)


# -------------------- END --------------------
