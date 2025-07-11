import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
st.caption("🆕 خيار تحميل النتائج بصيغة CSV")
st.markdown("ارفع ملف بلاغات (Excel)، وسيتم تحديد مستوى الأولوية تلقائيًا")

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
# 5) حساب الـ score والـ boost
# —————————————————————————————————————————————————————————————
df["score"] = (
    df["درجة الخطورة"] * weights["درجة الخطورة"]
    + df["عدد البلاغات"] * weights["عدد البلاغات"]
    + df["عدد السكان"] * weights["عدد السكان"]
    + df["صفة الموقع"] * weights["صفة الموقع"]
)
df["boost"] = df.apply(apply_contextual_boost, axis=1)
df["score"] += df["boost"]

# —————————————————————————————————————————————————————————————
# 6) التنبؤ وعرض النتائج
# —————————————————————————————————————————————————————————————
X_new = df[features]
df["مستوى_الأولوية"] = model.predict(X_new)

st.success("✅ تم التقييم! النتائج بالأسفل:")
st.dataframe(
    df[[
        "نوع الخدمة",
        "موقع البلاغ",
        "عدد السكان",
        "عدد البلاغات",
        "درجة الخطورة",
        "صفة الموقع",
        "مستوى_الأولوية",
    ]]
)

# —————————————————————————————————————————————————————————————
# 7) زر تحميل النتائج بصيغة CSV (بدون أي بافرات)
# —————————————————————————————————————————————————————————————
csv_data = df.to_csv(index=False)
st.download_button(
    label="⬇️ تحميل النتائج (CSV)",
    data=csv_data,
    file_name="نتائج_الأولوية.csv",
    mime="text/csv",
)
