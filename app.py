import streamlit as st
import pandas as pd
import joblib
import numpy as np

# —————————————————————————————————————————————————————————————
# 0) حقن CSS مخصَّص
# —————————————————————————————————————————————————————————————
# —————————————————————————————————————————————————————————————
# 0) إعداد الصفحة
# —————————————————————————————————————————————————————————————
st.set_page_config(page_title="سناد", layout="centered")

# —————————————————————————————————————————————————————————————
# 1) حقن Bootstrap و CSS مخصَّص (قبل أي مكوّن UI)
# —————————————————————————————————————————————————————————————
st.markdown(
    """
    <!-- Bootstrap CSS -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-ENjdO4Dr2bkBIFxQpeo5sJbL/w9iYhZ+3oQZl5Nq3uZMl+4xVZ9ZZTtmI3UksdQR"
      crossorigin="anonymous"
    >
    <style>
      /* خلفية الصفحة بيضاء ونص أسود */
      [data-testid="stAppViewContainer"] {
        background-color: #fff;
      }
      [data-testid="stAppViewContainer"] * {
        color: #000 !important;
      }

      /* Dropzone كامل أخضر */
      [data-testid="stFileUploaderDropzone"] {
        background-color: #28a745 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"],
      [data-testid="stFileUploaderDropzone"] small {
        color: #000 !important;
      }
      [data-testid="stFileUploaderDropzone"] svg {
        fill: #000 !important;
      }
      /* زرّ Browse داخل الصندوق */
      [data-testid="stBaseButton-secondary"] {
        background-color: #fff !important;
        color: #000 !important;
        border: 1px solid #000 !important;
        font-weight: bold !important;
        border-radius: 4px !important;
      }

      /* أزرار Download وغيرها خضراء */
      .stButton>button,
      [data-testid="stDownloadButton"] button {
        background-color: #28a745 !important;
        color: #fff !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        padding: 0.6rem 1.2rem !important;
      }

      /* عناوين مركزيّة */
      h1, h2 {
        text-align: center !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# —————————————————————————————————————————————————————————————
# 2) شعار "سناد" مُتمركز
# —————————————————————————————————————————————————————————————
# تأكدي من رفع logo.png في جذر المشروع
st.markdown(
    """
    <div class="text-center mb-3">
      <img src="logo.png" width="150" alt="سناد Logo"/>
    </div>
    """,
    unsafe_allow_html=True
)

# —————————————————————————————————————————————————————————————
# 3) العنوان والوصف
# —————————————————————————————————————————————————————————————
st.markdown("<h1 class='fw-bold'>سناد</h1>", unsafe_allow_html=True)
st.caption("تحميل النتائج بصيغة CSV فقط")


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
# 2) دوال مساعدة كما كانت
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
st.set_page_config(page_title="سناد", layout="centered")
st.title("سناد")
st.caption("تحميل النتائج بصيغة CSV فقط")

uploaded_file = st.file_uploader(" حمّل ملف البلاغلات", type=["xlsx"])
if not uploaded_file:
    st.info("لم يتم رفع ملف بعد")
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

st.success("تم التقييم:")

# بدل st.dataframe استخدم جدول HTML مع كلاسات Bootstrap
html_table = (
    df.head(10)[[
        "نوع الخدمة","موقع البلاغ","عدد السكان",
        "عدد البلاغات","درجة الخطورة","صفة الموقع",
        "مستوى_الأولوية"
    ]]
    .to_html(index=False, classes="table table-dark table-striped")
)
st.markdown(html_table, unsafe_allow_html=True)


# —————————————————————————————————————————————————————————————
# 7) زر تحميل النتائج بصيغة CSV
# —————————————————————————————————————————————————————————————
csv_data = df.to_csv(index=False)
st.download_button(
    label=" تحميل النتائج (CSV)",
    data=csv_data,
    file_name="نتائج_الأولوية.csv",
    mime="text/csv"
)

