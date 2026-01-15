UCI Default of Credit Card Clients — 資料清理說明
--------------------------------------------------
此資料集用於信用風險模型，目標為預測客戶下個月是否發生信用卡違約。
原始資料共 30,000 筆，經清理後移除 35 筆重複資料，最終保留 29,965 筆樣本。
目標欄位為「default payment next month」（1 = 違約，0 = 正常）。

[主要處理步驟]
1. 移除 ID 欄位（僅為識別用，與違約行為無關）。
2. 修正 EDUCATION (0,5,6→4)；MARRIAGE (0→3)：將非法值 0、5、6 合併為 4（其他類別）。
3. 將 SEX、EDUCATION、MARRIAGE 轉為類別型態（category），以利後續 one-hot 編碼。
4. PAY_0～PAY_6 欄位中出現的 -2 值，代表「該期無應付款或帳單已結清」，故合併為 -1（視為按時付款或無需付款）。
5. 新增衍生特徵（Derived Features）：
   - CREDIT_UTILIZATION：最近一期信用使用率（BILL_AMT1 / LIMIT_BAL，截斷於 [0,1]）。
   - BILL_CHANGE_MEAN：六期帳單變化程度（相鄰差的平均絕對值）。
   - MAX_DELAY：最嚴重延遲月數（PAY_0~PAY_6 的最大值）。
   - LAST_DELAY：最近一期延遲月數（PAY_0）。
   - RECENT_PAY_RATIO：最近一期繳款比例（PAY_AMT1 / BILL_AMT1，無帳單時為 0）。
   - PAYMENT_STD：六期繳款金額的標準差（繳款一致性）。
   - TOTAL_BILL：六期帳單總額。
   - TOTAL_PAY：六期繳款總額。
   - PAY_TO_BILL_RATIO：長期繳清率（TOTAL_PAY / TOTAL_BILL，無帳單時為 0）。
6. 標準化數值欄位（建立 scaled 版本）（適用 Logistic Regression、SVM、Neural Network）。

[輸出檔案]
- uci_default_cleaned.csv：未標準化版本（適用於 Random Forest、XGBoost、CatBoost 等樹模型）。
- uci_default_cleaned_scaled.csv：標準化版本（適用於 Logistic Regression、SVM、Neural Network）。
- uci_default_schema.csv：欄位型態與描述（含原始欄位與 [Derived] 衍生特徵）。

[欄位說明]
以下依據 Yeh & Lien (2009) 定義與本研究新增特徵進行整理；標記為 [Derived] 者為本研究衍生特徵：
- LIMIT_BAL: 給定信用額度（包含個人與家庭額度），單位：新台幣
- SEX: 性別（1 = 男性，2 = 女性）
- EDUCATION: 教育程度（1 = 研究所，2 = 大學，3 = 高中，4 = 其他）
- MARRIAGE: 婚姻狀況（1 = 已婚，2 = 單身，3 = 其他）
- AGE: 年齡（年）
- PAY_0: Sep 2005 還款狀態 (-1 = 準時付款，0 = 無延遲，1~9 = 延遲月數)
- PAY_2: Aug 2005 還款狀態
- PAY_3: Jul 2005 還款狀態
- PAY_4: Jun 2005 還款狀態
- PAY_5: May 2005 還款狀態
- PAY_6: Apr 2005 還款狀態
- BILL_AMT1: Sep 2005 帳單金額（單位：新台幣）
- BILL_AMT2: Aug 2005 帳單金額
- BILL_AMT3: Jul 2005 帳單金額
- BILL_AMT4: Jun 2005 帳單金額
- BILL_AMT5: May 2005 帳單金額
- BILL_AMT6: Apr 2005 帳單金額
- PAY_AMT1: Sep 2005 實際繳款金額（單位：新台幣）
- PAY_AMT2: Aug 2005 實際繳款金額
- PAY_AMT3: Jul 2005 實際繳款金額
- PAY_AMT4: Jun 2005 實際繳款金額
- PAY_AMT5: May 2005 實際繳款金額
- PAY_AMT6: Apr 2005 實際繳款金額
- default payment next month: 是否在下個月發生違約（1 = 違約，0 = 正常）
-  CREDIT_UTILIZATION: [Derived] 信用使用率（最近一期）：BILL_AMT1 / LIMIT_BAL，截斷於 [0,1]（0 = 未使用額度，1 = 額度全滿）
-  BILL_CHANGE_MEAN: [Derived] 帳單變化程度：六期相鄰帳單差的平均絕對值（數值越大 = 消費波動越大）
-  MAX_DELAY: [Derived] 最嚴重延遲月數：PAY_0~PAY_6 的最大值（-1 = 準時或無應付款；0 = 無延遲；1~9 = 延遲 N 月）
-  LAST_DELAY: [Derived] 最近一期延遲月數：PAY_0（-1 = 準時或無應付款；0 = 無延遲；1~9 = 延遲 N 月）
-  RECENT_PAY_RATIO: [Derived] 最近一期繳款比例：PAY_AMT1 / BILL_AMT1，無帳單則為 0（>1 = 溢繳）
-  PAYMENT_STD: [Derived] 繳款一致性：六期繳款金額的標準差（數值越高 = 繳款波動越大）
-  TOTAL_BILL: [Derived] 六期帳單總額（BILL_AMT1~6 加總）（反映近半年消費總量）
-  TOTAL_PAY: [Derived] 六期繳款總額（PAY_AMT1~6 加總）（反映近半年還款總額）
-  PAY_TO_BILL_RATIO: [Derived] 長期繳清率：TOTAL_PAY / TOTAL_BILL，無帳單則為 0（>1 = 長期溢繳）

所有 derived features 均經過 Winsorization（1% ~ 99%）檢查，無極端值異常。

---------- file data ----------
- team
  - csv
    - default of credit card clients.csv：原始官網下載dataset, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    - uci_default_cleaned.csv：已整理, 未標準化版本（適用於 Random Forest、XGBoost、CatBoost 等樹模型）。
    - uci_default_cleaned_scaled.csv：已整理, 標準化版本（適用於 Logistic Regression、SVM、Neural Network）。
    - uci_default_schema.csv：欄位型態與描述（含原始欄位與 [Derived] 衍生特徵）。
  - Report
    - Data_Preprocessing_EDA.docx：報告 Word 檔
    - Data_Preprocessing_EDA.pdf：報告 PDF 檔
  - uci_default_README.txt：README 說明文件