import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# 1. 載入資料
# 注意：Logistic Regression 需要標準化後的資料 (scaled)，樹模型通常不需要
# 為了簡化流程，我們統一使用「原始資料」訓練樹模型，
# 並在程式中為 Logistic Regression 建立一個 Pipeline (自動包含標準化)
file_path = 'uci_default_cleaned.csv' 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_path}")
    exit()

target = 'default payment next month'
X = df.drop(columns=[target])
y = df[target]

# 2. 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 模型 1: Logistic Regression (LR)
# ---------------------------------------------------------
print("正在訓練 Logistic Regression...")
from sklearn.pipeline import Pipeline
# LR 對特徵縮放敏感，所以我們用 Pipeline 把 StandardScaler 包進去
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'lr_model.pkl')
print("-> 已儲存 lr_model.pkl")

# ---------------------------------------------------------
# 模型 2: Random Forest (RF)
# ---------------------------------------------------------
print("正在訓練 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10,
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.pkl')
print("-> 已儲存 rf_model.pkl")

# ---------------------------------------------------------
# 模型 3: XGBoost (XGB)
# ---------------------------------------------------------
print("正在訓練 XGBoost...")
# XGBoost 的 scale_pos_weight 用來處理不平衡 (類似 class_weight)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1) #
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.pkl')
print("-> 已儲存 xgb_model.pkl")

# ---------------------------------------------------------
# 儲存共用資料
# ---------------------------------------------------------
# 儲存特徵名稱
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

# 儲存參考數據 (用於同儕比較)
reference_data = X_test.sample(n=1000, random_state=42)
reference_data.to_csv('reference_data.csv', index=False)

print("\n全部完成！所有模型已儲存。")