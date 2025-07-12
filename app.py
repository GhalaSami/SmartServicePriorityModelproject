import streamlit as st
import pandas as pd
import joblib
import numpy as np

# —————————————————————————————————————————————————————————————
# 0) تخصيص المظهر عبر CSS
# —————————————————————————————————————————————————————————————
st.markdown(
    """
    <style>
    /* خلفية الموقع بيضاء */
    .reportview-container, .main {
        background-color: #FFFFFF;
    }
    /* عنوان الصفحة */
    .stTitle {
        text-align: center !important;
        color: #000000;
        font-weight: bold;
    }
    /* الزرّات خضراء ونصها أبيض */
    .stButton>button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold !important;
        border: none;
        border-radius: 5px;
        padding: 0.6em 1.2em;
    }
    /* ترويسة الجدول ونص الخلايا */
    table.dataframe th, table.dataframe td {
        border: 1px solid #ddd;
        text-align: center;       /* محاذاة النص في الوسط */
        color: #000000;           /* خط غامق أسود */
        font-weight: normal;
        direction: rtl;           /* لاتجاه النص العربي */
    }
    /* لتعزيز إطار الجدول */
    table.dataframe {
        border-collapse: collapse;
        width: 100%;
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# —————————————————————————————————————————————————————————————
# 1) تحميل المودل وكل Artefacts
# —————————————————————————————————————————————————————————————
artifacts = joblib.load("final_model.pkl")
model = artifacts["model"]
features = artifacts["features"]
danger_map = artifacts["danger_map"]
weights = artifacts["weights"]
group1_locations = artifacts["group1_locations"]
group2_locations = artifacts["group2_locations"]
group3_locations = artifacts["group3_locations"]
group1_classification = artifacts["group1_classification"]
group2_classification = artifacts["group2_classification"]
group3_classification = artifacts["group3_classification"]

# —————————————————————————————————————————————————————————————
# 2) دوال مساعدة
# —————————————————————————————————————————————————————————————
def classify_site_attribute(location):
    if location in group1_locations:
        return group1_classification
    elif location in group2_locations:
        return group2_classification
    elif location in group3_locations:
        return group3_classification
    else:
        return "غير محدد"

site_rank_map = {
    group2_classification: 3,
    group1_classification: 2,
    group3_classification: 1,
    "غير محدد": 0
}

def apply_contextual_boost(row):
    boost = 0
    if row["درجة الخطورة"] >= 4 and row["عدد البلاغات"] >= 3:
        boost += 5000
    if row["صفة الموقع"] == site_rank_map[group2_classification]:
        boost += 3000
    if row["عدد السكان"] > 100000 and row["درجة الخطورة"] >= 3:
        boost += 4000
    boost += np.random.randint(-100, 100)
    return boost

# —————————————————————————————————————————————————————————————
# 3) واجهة Streamlit
# —————————————————————————————————————————————————————————————
st.set_page_config(page_title="نموذج أولوية البلاغات", layout="centered")
st.title("📊 نظام تحديد أولوية البلاغات")
st.caption("🆕 الآن مع جدول مُحاط بإطار ونص وسط وCSV Download")
st.markdown("ارفع ملف بلاغات (Excel)، وسيتم تحديد مستوى الأولوية وتصديره بصيغة CSV")

uploaded_file = st.file_uploader("📁 حمّل ملف البلاغات (Excel)", type=["xlsx"])
if not uploaded_file:
    st.info("لم يتم رفع ملف بعد.")
    st.stop()

# —————————————————————————————————————————————————————————————
# 4) قراءة وتجهيز البيانات
# —————————————————————————————————————————————————————————————
df = pd.read_excel(uploaded_file)
if "عدد تكرار البلاغ" in df.columns:
    df = df.rename(columns={"عدد تكرار البلاغ": "عدد البلاغات"})

df["درجة الخطورة"] = df["نوع الخدمة"].map(danger_map)
df["صفة الموقع"] = df["موقع البلاغ"].apply(classify_site_attribute)
df["صفة الموقع"] = df["صفة الموقع"].map(site_rank_map)

# —————————————————————————————————————————————————————————————
# 5) حساب score و boost
# —————————————————————————————————————————————————————————————
df["score"] = (
    df["درجة الخطورة"] * weights["درجة الخطورة"]
  + df["عدد البلاغات"] * weights["عدد البلاغات"]
  + df["عدد السكان"]     * weights["عدد السكان"]
  + df["صفة الموقع"]      * weights["صفة الموقع"]
)
df["boost"] = df.apply(apply_contextual_boost, axis=1)
df["score"] += df["boost"]

# —————————————————————————————————————————————————————————————
# 6) التنبؤ وعرض النتائج
# —————————————————————————————————————————————————————————————
X_new = df[features]
df["مستوى_الأولوية"] = model.predict(X_new)

st.success("✅ تم التقييم! النتائج بالأسفل:")
# عرض الجدول بإطار ومحاذاة مركزية
st.dataframe(
    df[[
        "نوع الخدمة","موقع البلاغ","عدد السكان",
        "عدد البلاغات","درجة الخطورة","صفة الموقع",
        "مستوى_الأولوية"
    ]],
    use_container_width=True
)

# —————————————————————————————————————————————————————————————
# 7) زر تحميل النتائج بصيغة CSV
# —————————————————————————————————————————————————————————————
csv_data = df.to_csv(index=False)
st.download_button(
    label="⬇️ تحميل النتائج (CSV)",
    data=csv_data,
    file_name="نتائج_الأولوية.csv",
    mime="text/csv"
)
