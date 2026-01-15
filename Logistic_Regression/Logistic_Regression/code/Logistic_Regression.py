# 步驟 1: 導入必要的函式庫
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# 假設您的 CSV 檔案存放在專案根目錄下的 'data/' 資料夾中
# 步驟 3: 載入資料 - 使用本地相對路徑
file_path = './data/uci_default_cleaned_scaled.csv'

try:
    # 修正：加入 quoting 參數，告訴 Pandas 檔案中的數據欄位被雙引號包圍
    df = pd.read_csv(file_path, quoting=1)

    # 說明：quoting=1 等價於 csv.QUOTE_ALL，指示讀取器預期數據欄位被引號包圍。

except FileNotFoundError:
    print(f"錯誤：找不到檔案。請檢查路徑是否正確：{file_path}")
    exit()

print(f"資料載入成功，樣本數: {len(df)}")
print("數據前五行：")
print(df.head())

# 步驟 4: 類別變數編碼 (One-Hot Encoding)
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

# drop_first=True 可避免共線性問題，但對於基準模型 (Logistic Regression) 仍建議使用
# get_dummies 會將 'SEX' 欄位替換為 'SEX_1', 'SEX_2' 等虛擬變數
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
#將 df 中的 categorical_cols 所列的欄位轉換為多個虛擬變數（Dummy Variables）

# 步驟 5: 定義特徵 (X) 和目標變數 (y)
target_col = 'default payment next month' #目標變數
X = df.drop(columns=[target_col]) # 定義特徵矩陣 X（自變數）。從整個 DataFrame df 中移除目標變數欄位 target_col，剩下的就是所有的特徵。
y = df[target_col] #定義目標向量 y（應變數）。只保留 target_col 欄位，即客戶是否違約的標記。

# 步驟 6: 劃分訓練集與測試集
RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 常用 70% 訓練 / 30% 測試
    random_state=RANDOM_SEED,
    stratify=y # 使用 stratify 確保訓練集和測試集的違約比例一致
)

print("\n資料劃分完成。")
print(f"訓練集樣本數: {len(X_train)} | 測試集樣本數: {len(X_test)}")

# 步驟 7: 訓練羅吉斯迴歸基準模型
# 使用 class_weight='balanced' 處理資料不平衡
# 由於違約（1）是少數類別，此參數會自動調整類別權重，讓模型更重視對少數類別的預測，以提高召回率。
log_reg_baseline = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    random_state=RANDOM_SEED,
    max_iter=1000 # 確保模型收斂
)

print("\n開始訓練羅吉斯迴歸模型...")
log_reg_baseline.fit(X_train, y_train)
print("模型訓練完成。")

# 步驟 8: 預測與效能評估
# 預測違約機率 (用於計算 AUC)
y_pred_proba = log_reg_baseline.predict_proba(X_test)[:, 1]

# 預測類別標籤 (用於計算 Recall/Accuracy)
y_pred = log_reg_baseline.predict(X_test)

# 計算評估指標
auc_score = roc_auc_score(y_test, y_pred_proba) # 專案要求的整體判別力指標
recall = recall_score(y_test, y_pred) # 召回率 (捕捉違約客戶的能力):衡量在所有實際違約客戶中，模型成功預測出違約的比例。
accuracy = accuracy_score(y_test, y_pred) # 準確率:衡量模型預測正確的樣本佔總樣本的比例。

print("\n--- 基準模型效能評估 (測試集) ---")
print(f"AUC (Area Under the Curve): {auc_score:.4f} (判斷力指標)")
print(f"召回率 (Recall Score):     {recall:.4f} (捕捉違約客戶能力)")
print(f"準確率 (Accuracy Score):  {accuracy:.4f}")
print("\n分類報告 (Classification Report):")
print(classification_report(y_test, y_pred)) # 輸出詳細的分類報告，包含 F1-score、精確率（Precision）、召回率（Recall）指標

# 步驟 9: 係數分析 (模型解釋)
print("\n--- 羅吉斯迴歸係數分析 (特徵影響力與風險方向) ---")

# 獲取係數和特徵名稱
coefficients = log_reg_baseline.coef_[0] #從訓練好的模型中提取所有特徵的係數（w1,w2,..）。係數代表每個特徵對違約風險的影響程度。
feature_names = X_train.columns

# 創建係數 DataFrame 並計算絕對值
# 將特徵名稱和對應的係數組織成一個 DataFrame，方便後續排序和分析
# 計算係數的絕對值。因為影響力（強度）只看絕對值大小，正負號只代表方向。
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# 輸出前 15 大影響特徵
print(f"截距 (Intercept): {log_reg_baseline.intercept_[0]:.4f}")
print("\n前 15 大影響特徵：")
print("（係數的絕對值越大，影響力越大。正值 (+) 表示該特徵增加違約風險，負值 (-) 表示降低違約風險）")

for index, row in coef_df.head(15).iterrows():
    # 根據係數判斷風險方向
    sign = '增加違約風險 (+)' if row['Coefficient'] > 0 else '降低違約風險 (-)'

    # 處理 One-Hot 編碼後的特徵名稱
    feature_display = row['Feature']
    if 'SEX_' in feature_display:
        feature_display = feature_display.replace('SEX_1', 'SEX (男性)')\
                                         .replace('SEX_2', 'SEX (女性)')
    elif 'EDUCATION_' in feature_display:
        feature_display = feature_display.replace('EDUCATION_1', 'EDUCATION (研究所)')\
                                         .replace('EDUCATION_2', 'EDUCATION (大學)')\
                                         .replace('EDUCATION_3', 'EDUCATION (高中)')\
                                         .replace('EDUCATION_4', 'EDUCATION (其他)')
    elif 'MARRIAGE_' in feature_display:
        feature_display = feature_display.replace('MARRIAGE_1', 'MARRIAGE (已婚)')\
                                         .replace('MARRIAGE_2', 'MARRIAGE (單身)')\
                                         .replace('MARRIAGE_3', 'MARRIAGE (其他)')

    print(f"  {feature_display:<30}: {row['Coefficient']:.4f} ({sign})")

print("\n--- 對整個資料集進行違約機率預測 ---")
# predict_proba() 返回 (P(class 0), P(class 1)) 的機率陣列
all_proba = log_reg_baseline.predict_proba(X)
# 取出違約 (Class 1) 的預測機率
predicted_prob_default = all_proba[:, 1]

# 步驟 10: 整合結果並生成最終 CSV
# 1. 建立結果 DataFrame
# 假設 DataFrame 的 Index 可以作為客戶識別 (ID)
final_output_df = pd.DataFrame({
    'Client_Index': df.index,  # 使用 DataFrame 的 Index 作為客戶 ID (如果原始資料沒有 ID 欄位)
    'Actual_Default_Label': y, # 實際的違約標籤 (0 或 1)
    'Predicted_Prob_Default': predicted_prob_default # 羅吉斯迴歸預測的違約機率 (0.00 ~ 1.00)
})

# 2.將機率轉換為一個簡單的風險評分 (例如 0~100 分)
# 這裡使用簡單的反轉機率轉換為風險分數 (分數越高，風險越低)
print(f"最終結果 DataFrame 建立完成，樣本數: {len(final_output_df)}")


# 步驟 11: 儲存最終 CSV 檔案 - 使用本地相對路徑
output_csv_path = './data/Model1_Baseline_Final_All_Predictions.csv'
final_output_df.to_csv(output_csv_path, index=False)

print(f"\n--- 最終違約機率檔案已成功儲存至: {output_csv_path} ---")

