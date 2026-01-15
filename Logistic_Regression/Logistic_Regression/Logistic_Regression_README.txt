README.txt - 羅吉斯迴歸基準模型 (Model 1) 

==================================================
專題名稱：個人信用風險評分模型
模型名稱：羅吉斯迴歸 (Logistic Regression) 基準模型
程式檔案：Logistic_Regression.py
作者：黎亭君 (Model 1 負責人)
日期：2025/11/16
==================================================

一、 環境準備與檔案配置
--------------------------------------------------


2.  建立專案資料夾結構：
   
    Credit_Risk_Project/
    ├── data/
    │   └── uci_default_cleaned_scaled.csv  <-- 【數據檔案】
    └── Logistic_Regression.py             <-- 【程式碼檔案】



二、 程式執行過程說明 (步驟解析)
--------------------------------------------------

程式運行將自動完成以下所有步驟：

| 步驟 | 程式碼區塊 | 執行動作與目的 |
| :--- | :--- | :--- |
| **Step 1-3** | 數據載入 (Loading) | 導入函式庫，並使用 `./data/uci_default_cleaned_scaled.csv` 載入數據。使用 `quoting=1` 確保 CSV 檔案被正確解析。 |
| **Step 4** | 數據預處理 (Preprocessing) | 對 `SEX`, `EDUCATION`, `MARRIAGE` 三個類別變數進行 **One-Hot Encoding** (獨熱編碼)，將其轉換為羅吉斯迴歸可處理的數值特徵。 |
| **Step 5** | 數據劃分 (Splitting) | 將數據按 $70\%$ (訓練集) / $30\%$ (測試集) 劃分，並使用 **`stratify=y`** 參數確保訓練集和測試集的違約比例一致 (約 $22.12\%$)。 |
| **Step 6-7** | 模型訓練 (Training) | 建立 `LogisticRegression` 模型，並設定 **`class_weight='balanced'`** 參數，以提高模型對少數類別（違約客戶）的捕捉能力 (召回率)。 |
| **Step 8** | 效能評估 (Evaluation) | 對測試集 (X_test) 進行預測，並輸出關鍵指標：**AUC** (整體判別力)、**Recall Score** (違約捕捉能力) 和詳細的 **Classification Report** (分類報告)。 |
| **Step 9** | 係數分析 (Interpretation) | 提取模型係數，找出前 15 大影響特徵。這一步提供了**業務可解釋性**，解釋了哪些因素是增加或降低違約風險的最重要因子。 |
| **Step 10-11** | 數據輸出 (Output) | 對所有 $29,965$ 筆客戶數據進行預測，生成一個包含客戶 ID、實際標籤和預測機率的 `final_output_df`。該結果儲存為 `./data/Model1_Baseline_Final_All_Predictions.csv` 檔案。 |

三、 最終輸出檔案位置
--------------------------------------------------

執行成功後，您將在 **`Credit_Risk_Project/data/`** 資料夾中找到最終的數據成果檔案：

* **`Model1_Baseline_Final_All_Predictions.csv`**：包含所有客戶的羅吉斯迴歸預測違約機率。

