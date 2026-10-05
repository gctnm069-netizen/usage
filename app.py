import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# ページ設定
st.set_page_config(
    page_title="エネルギー消費量 予測ダッシュボード",
    layout="wide"
)

# スプレッドシートURL
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1lKGfQNYi2a5R9c9vIYsAFzRswe43ERB9xiULpQwdqqM/edit?usp=drive_link"

# ---------------------------------------------------------
# 熱量換算係数（一次エネルギー換算 → 千kcal）
# ---------------------------------------------------------
ELEC_COEF = 0.8604     # 電気 (kWh → 千kcal)
GAS_COEF = 10.755      # ガス (m³ → 千kcal)
OIL_COEF = 9.3449      # 重油 (L → 千kcal)

# ---------------------------------------------------------
# Google スプレッドシート連携
# ---------------------------------------------------------
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data_from_gsheets():
    try:
        df = conn.read(spreadsheet=SPREADSHEET_URL, ttl="10s")

        required_cols = ['year', 'month', 'temp', 'electricity', 'gas', 'heavy_oil', 'total']
        if not all(col in df.columns for col in required_cols) or df.empty:
            return generate_default_data()

        # 🔥 電気とガスを ×1000 補正
        df['electricity'] = df['electricity'] * 1000
        df['gas'] = df['gas'] * 1000

        # 🔥 総合熱量を再計算（スプレッドシートの total は使わない）
        df['total'] = (
            df['electricity'] * ELEC_COEF +
            df['gas'] * GAS_COEF +
            df['heavy_oil'] * OIL_COEF
        )

        return df

    except:
        st.warning("Google スプレッドシート読み込み失敗 → 初期データを使用します")
        return generate_default_data()

def generate_default_data():
    """2021年4月〜2027年3月（72ヶ月分）のデフォルトデータ生成"""
    np.random.seed(42)
    start_date = pd.date_range(start="2021-04-01", periods=72, freq="MS")
    years = start_date.year.values
    months = start_date.month.values

    base_temps = np.array([15.2, 19.8, 23.5, 27.8, 29.1, 24.6, 18.5, 13.1, 8.4, 6.2, 7.1, 10.5])
    temps = np.tile(base_temps, 6) + np.random.normal(0, 0.8, 72)
    temps = np.round(temps, 1)

    # 元のモデルで生成
    electricity = 350 - 15 * temps + 0.7 * (temps ** 2) + np.random.normal(0, 10, len(temps))
    gas = 120 - 4.5 * temps + 0.1 * (temps ** 2) + np.random.normal(0, 3, len(temps))
    heavy_oil = np.maximum(10, 500 - 18 * temps + np.random.normal(0, 15, len(temps)))

    # 🔥 電気とガスを ×1000 補正
    electricity *= 1000
    gas *= 1000

    # 🔥 熱量換算（千kcal）
    total = (
        electricity * ELEC_COEF +
        gas * GAS_COEF +
        heavy_oil * OIL_COEF
    )

    return pd.DataFrame({
        'year': years,
        'month': months,
        'temp': temps,
        'electricity': electricity,
        'gas': gas,
        'heavy_oil': heavy_oil,
        'total': total
    })

# 初期データ読み込み
if 'df_data' not in st.session_state:
    st.session_state['df_data'] = load_data_from_gsheets()

# ---------------------------------------------------------
# サイドバー
# ---------------------------------------------------------
st.sidebar.title("📌 メニュー")

page = st.sidebar.radio(
    "画面を選択してください:",
    ["📊 予測ダッシュボード", "📝 過去データ入力・編集"]
)

st.sidebar.markdown("---")

# ---------------------------------------------------------
# PAGE 1: データ編集
# ---------------------------------------------------------
if page == "📝 過去データ入力・編集":
    st.title("📝 過去実績データの登録・編集")

    if st.button("🔄 スプレッドシートから最新データを再読み込み"):
        st.session_state['df_data'] = load_data_from_gsheets()
        st.rerun()

    edited_df = st.data_editor(
        st.session_state['df_data'],
        num_rows="dynamic",
        use_container_width=True,
        height=500
    )

    if st.button("💾 Googleスプレッドシートへ保存", type="primary"):
        try:
            conn.update(spreadsheet=SPREADSHEET_URL, data=edited_df)
            st.session_state['df_data'] = edited_df
            st.success("保存完了！")
        except:
            st.session_state['df_data'] = edited_df
            st.warning("スプレッドシート書き込み権限を確認してください")

# ---------------------------------------------------------
# PAGE 2: 予測ダッシュボード
# ---------------------------------------------------------
else:
    df = st.session_state['df_data'].copy()
    df['year_month'] = df.apply(lambda r: f"{int(r['year'])}/{int(r['month']):02d}", axis=1)

    st.sidebar.title("⚙️ 条件設定")

    energy_option = st.sidebar.selectbox(
        "表示するエネルギー種別:",
        ["電気", "ガス", "重油", "総合"]
    )

    energy_map = {
        "電気": {"col": "electricity", "unit": "kWh", "degree": 2},
        "ガス": {"col": "gas", "unit": "m³", "degree": 2},
        "重油": {"col": "heavy_oil", "unit": "L", "degree": 1},
        "総合": {"col": "total", "unit": "千kcal", "degree": 2}
    }

    target = energy_map[energy_option]
    target_col = target["col"]
    unit = target["unit"]
    degree = target["degree"]

    input_temp = st.sidebar.slider(
        "翌月の予想平均気温 (°C)",
        min_value=-5.0, max_value=38.0, value=30.5, step=0.5
    )

    # モデル構築
    X = df[['temp']]
    y = df[target_col]
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X, y)

    predicted_value = model.predict(pd.DataFrame([[input_temp]], columns=['temp']))[0]

    st.title(f"📊 エネルギー需要予測ダッシュボード（{energy_option}）")

    col1, col2, col3 = st.columns(3)
    col1.metric("翌月想定気温", f"{input_temp:.1f} ℃")
    col2.metric(f"翌月予測消費量 ({energy_option})", f"{predicted_value:,.1f} {unit}")
    diff = predicted_value - y.mean()
    col3.metric("過去平均との差分", f"{diff:+,.1f} {unit}", delta=f"{diff:.1f} {unit}")

    st.markdown("---")

    # 回帰曲線
    temp_values = np.linspace(X['temp'].min() - 3, X['temp'].max() + 3, 100)
    predicted_curve = model.predict(pd.DataFrame(temp_values, columns=['temp']))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X['temp'], y, alpha=0.7, label="Actual")
    ax.plot(temp_values, predicted_curve, linestyle='--', color='red', label="Model")
    ax.scatter(input_temp, predicted_value, color='green', s=120, label="Prediction")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(f"{energy_option} ({unit})")
    ax.legend()
    ax.grid(True, linestyle=':')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📋 過去実績データ")
    st.dataframe(
        df[['year_month', 'temp', target_col]].rename(
            columns={'year_month': '年月', 'temp': '気温(℃)', target_col: f'消費量({unit})'}
        ),
        height=380
    )

    if energy_option == "総合":
        st.markdown("---")
        st.subheader("🌐 エネルギー種別の比較")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df['temp'], df['electricity'], 'o-', label='Electricity (kWh)')
        ax2.plot(df['temp'], df['gas'], 's-', label='Gas (m³)')
        ax2.plot(df['temp'], df['heavy_oil'], '^--', label='Heavy Oil (L)')
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Raw Units")
        ax2.legend()
        ax2.grid(True, linestyle=':')
        st.pyplot(fig2)
        plt.close(fig2)
