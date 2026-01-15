import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# 設定頁面資訊
st.set_page_config(page_title="個人信用風險評估系統", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 設定資料夾路徑
# =========================================
# 設定存放模型與數據的子資料夾名稱，請確保你的資料夾叫 'data'
DATA_DIR = 'data' 

# 輔助函式：取得檔案的完整路徑
def get_path(filename):
    return os.path.join(DATA_DIR, filename)

# =========================================
# 0. 特徵中文化字典
# =========================================
feature_map = {
    'LIMIT_BAL': '信用額度',
    'SEX': '性別',
    'EDUCATION': '教育程度',
    'MARRIAGE': '婚姻狀況',
    'AGE': '年齡',
    'PAY_0': '上個月還款狀態',
    'PAY_2': '兩個月前還款狀態',
    'PAY_3': '三個月前還款狀態',
    'PAY_4': '四個月前還款狀態',
    'PAY_5': '五個月前還款狀態',
    'PAY_6': '六個月前還款狀態',
    'BILL_AMT1': '最近一期帳單金額',
    'BILL_AMT2': '兩個月前帳單金額',
    'PAY_AMT1': '最近一期繳款金額',
    'PAY_AMT2': '兩個月前繳款金額',
    'CREDIT_UTILIZATION': '信用額度使用率',
    'MAX_DELAY': '歷史最長延遲月數',
    'LAST_DELAY': '最近一次延遲月數',
    'BILL_CHANGE_MEAN': '帳單變動幅度',
    'RECENT_PAY_RATIO': '近期繳款比例',
    'PAYMENT_STD': '繳款金額穩定度',
    'TOTAL_BILL': '總帳單金額',
    'TOTAL_PAY': '總繳款金額',
    'PAY_TO_BILL_RATIO': '總繳款/總帳單比例'
}

def get_cn_name(eng_name):
    return f"{feature_map.get(eng_name, eng_name)}"

# =========================================
# 1. 載入資源 (修改為使用 get_path)
# =========================================
@st.cache_data
def load_common_data():
    try:
        # 使用 get_path 指向 data 資料夾
        feature_names = joblib.load(get_path('feature_names.pkl'))
        ref_data = pd.read_csv(get_path('reference_data.csv'))
        return feature_names, ref_data
    except FileNotFoundError:
        return None, None

def load_model(model_name):
    try:
        # 使用 get_path 指向 data 資料夾
        if model_name == "Random Forest":
            return joblib.load(get_path('rf_model.pkl'))
        elif model_name == "Logistic Regression":
            return joblib.load(get_path('lr_model.pkl'))
        elif model_name == "XGBoost":
            return joblib.load(get_path('xgb_model.pkl'))
    except FileNotFoundError:
        return None

feature_names, ref_data = load_common_data()

if feature_names is None:
    st.error(f"找不到基礎檔案！請確認 'feature_names.pkl' 與 'reference_data.csv' 是否已放入 '{DATA_DIR}' 資料夾中。")
    st.stop()

# =========================================
# 2. 側邊欄：設定與輸入
# =========================================
st.sidebar.title("🔧 設定與輸入")

# 模型選擇
st.sidebar.subheader("1. 選擇預測模型")
model_option = st.sidebar.selectbox(
    "請選擇要使用的演算法：",
    ("Random Forest", "Logistic Regression", "XGBoost")
)

current_model = load_model(model_option)
if current_model is None:
    st.error(f"找不到 {model_option} 的模型檔案！")
    st.stop()

st.sidebar.success(f"目前引擎: {model_option}")
st.sidebar.markdown("---")

# 使用者資料輸入
st.sidebar.subheader("2. 申請人資料輸入")

def user_input_features():
    limit_bal = st.sidebar.number_input("信用額度 (LIMIT_BAL)", 10000, 1000000, 150000, step=10000)
    age = st.sidebar.slider("年齡 (AGE)", 20, 80, 30)
    sex = st.sidebar.selectbox("性別", [1, 2], format_func=lambda x: "男" if x==1 else "女")
    education = st.sidebar.selectbox("教育", [1, 2, 3, 4], format_func=lambda x: {1:"研究所", 2:"大學", 3:"高中", 4:"其他"}[x])
    marriage = st.sidebar.selectbox("婚姻", [1, 2, 3], format_func=lambda x: {1:"已婚", 2:"單身", 3:"其他"}[x])
    
    # --- [關鍵修正] 還款狀態輸入範圍 ---
    st.sidebar.markdown("---")
    st.sidebar.caption("📋 **還款狀態說明**：")
    st.sidebar.caption(" **-1**: 準時還款 / 無消費 (已結清)")
    st.sidebar.caption(" **0**: 使用循環信貸 (有還最低)")
    st.sidebar.caption(" **1~9**: 延遲 1~9 個月")
    
    # 修正範圍：從 -1 到 9 (根據資料清理邏輯，-2 已被合併至 -1)
    pay_0 = st.sidebar.slider("上月還款 (PAY_0)", -1, 9, 0)
    pay_2 = st.sidebar.slider("兩月前 (PAY_2)", -1, 9, 0)
    
    bill_amt1 = st.sidebar.number_input("上期帳單金額", 0, 500000, 5000)
    pay_amt1 = st.sidebar.number_input("上期繳款金額", 0, 500000, 2000)

    # 模擬其他欄位 (簡化 Demo)
    data = {
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt1, 'BILL_AMT3': bill_amt1,
        'BILL_AMT4': bill_amt1, 'BILL_AMT5': bill_amt1, 'BILL_AMT6': bill_amt1,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt1, 'PAY_AMT3': pay_amt1,
        'PAY_AMT4': pay_amt1, 'PAY_AMT5': pay_amt1, 'PAY_AMT6': pay_amt1
    }
    
    # 自動計算衍生特徵
    data['CREDIT_UTILIZATION'] = data['BILL_AMT1'] / data['LIMIT_BAL'] if data['LIMIT_BAL'] > 0 else 0
    data['BILL_CHANGE_MEAN'] = 1000
    data['MAX_DELAY'] = max(pay_0, pay_2)
    data['LAST_DELAY'] = pay_0
    data['RECENT_PAY_RATIO'] = data['PAY_AMT1'] / data['BILL_AMT1'] if data['BILL_AMT1'] > 0 else 1
    data['PAYMENT_STD'] = 500
    data['TOTAL_BILL'] = bill_amt1 * 6
    data['TOTAL_PAY'] = pay_amt1 * 6
    data['PAY_TO_BILL_RATIO'] = data['TOTAL_PAY'] / data['TOTAL_BILL'] if data['TOTAL_BILL'] > 0 else 1
    
    df_input = pd.DataFrame(data, index=[0])
    return df_input[feature_names]

input_df = user_input_features()

# =========================================
# 3. 主畫面顯示
# =========================================
st.title("🏦 智慧信用風險評估儀表板")

if st.button('開始評估 (AI Diagnosis)'):
    
    # 1. 預測
    prob_default = current_model.predict_proba(input_df)[0][1]
    risk_score = int(prob_default * 100)

    if risk_score < 20:
        level, color = "低風險 (Low Risk)", "green"
    elif risk_score < 50:
        level, color = "中風險 (Medium Risk)", "orange"
    else:
        level, color = "高風險 (High Risk)", "red"

    # 2. 顯示指標
    c1, c2, c3 = st.columns(3)
    c1.metric("違約機率", f"{prob_default:.1%}")
    c2.metric("風險評分 (0-100)", f"{risk_score}")
    c3.markdown(f"#### <span style='color:{color}'>{level}</span>", unsafe_allow_html=True)
    
    st.divider()

    # 3. 圖表區
    tab1, tab2 = st.tabs(["📊 同儕比較", "🔍 風險因子瀑布圖 (AI 解釋)"])
    
    with tab1:
        st.subheader("您在群體中的風險位置")
        ref_probs = current_model.predict_proba(ref_data)[:, 1] * 100
        fig = px.histogram(x=ref_probs, nbins=20, labels={'x':'風險分數 (Risk Score)', 'y':'人數'}, title="風險分佈圖", opacity=0.7)
        fig.add_vline(x=risk_score, line_color="red", line_dash="dash", annotation_text="You")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"{model_option} 的決策解釋")
        
        try:
            # 準備 SHAP Values
            features = []
            values = []
            
            if model_option == "Logistic Regression":
                st.info("Logistic Regression 使用線性權重進行解釋：")
                if hasattr(current_model, 'named_steps'):
                    clf = current_model.named_steps['clf']
                else:
                    clf = current_model
                
                coeffs = clf.coef_[0]
                feat_imp = pd.DataFrame({'Feature': feature_names, 'Weight': coeffs})
                feat_imp['AbsWeight'] = feat_imp['Weight'].abs()
                top_feats = feat_imp.sort_values('AbsWeight', ascending=False).head(8)
                
                top_feats = top_feats.iloc[::-1]
                features = [get_cn_name(f) for f in top_feats['Feature']]
                values = top_feats['Weight'].tolist()

            else:
                # Tree Models
                explainer = shap.TreeExplainer(current_model)
                shap_values = explainer.shap_values(input_df, check_additivity=False)
                
                sv = None
                if isinstance(shap_values, list): # RF
                    sv = shap_values[1][0]
                elif len(np.array(shap_values).shape) == 3: # XGB 3D
                     sv = shap_values[0, :, 1]
                elif len(np.array(shap_values).shape) == 2: # XGB 2D
                    sv = shap_values[0]
                else:
                    sv = shap_values

                explanation_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': sv})
                explanation_df['AbsVal'] = explanation_df['SHAP Value'].abs()
                top_features = explanation_df.sort_values('AbsVal', ascending=False).head(8)
                
                top_features = top_features.iloc[::-1]
                features = [get_cn_name(f) for f in top_features['Feature']]
                values = top_features['SHAP Value'].tolist()

            # 繪製瀑布圖
            fig = go.Figure(go.Waterfall(
                name = "Risk Contribution",
                orientation = "h",
                measure = ["relative"] * len(values),
                y = features, 
                x = values,
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}},
                decreasing = {"marker":{"color":"#2ca02c"}}, 
                increasing = {"marker":{"color":"#d62728"}}, 
            ))

            fig.update_layout(
                title = "<b>風險評分構成 (紅色=增加風險 / 綠色=降低風險)</b>",
                showlegend = False,
                height=500,
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)

            # === AI 白話解釋生成 ===
            st.markdown("### 💡 AI 風險診斷報告")
            st.markdown("---")
            
            risk_df = pd.DataFrame({'Feature': features, 'Value': values})
            main_risk = risk_df[risk_df['Value'] > 0].sort_values('Value', ascending=False)
            
            if not main_risk.empty:
                top_risk_name = main_risk.iloc[0]['Feature']
                st.warning(f"⚠️ **主要扣分項目**：您的 **「{top_risk_name}」** 狀況較不理想。這通常代表有延遲還款紀錄、或是信用額度使用率過高，導致系統判定違約風險顯著增加。")
            else:
                st.success("✅ **表現優異**：目前沒有顯著的扣分項目，您的信用狀況非常健康！")

            main_advantage = risk_df[risk_df['Value'] < 0].sort_values('Value', ascending=True)
            
            if not main_advantage.empty:
                top_adv_name = main_advantage.iloc[0]['Feature']
                st.info(f"🛡️ **主要加分項目**：您的 **「{top_adv_name}」** 為您加分不少。這可能因為您的還款紀錄良好、或財務狀況穩定，有效降低了系統評估的風險。")

        except Exception as e:
            st.error(f"解釋圖表生成失敗: {e}")