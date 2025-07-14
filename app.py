import streamlit as st
import pandas as pd
import joblib
import numpy as np

# —————————————————————————————————————————————————————————————
# 0) Page configuration
# —————————————————————————————————————————————————————————————
st.set_page_config(page_title="سناد", layout="centered")
# —————————————————————————————————————————————————————————————
# 1) Display logo centered
# —————————————————————————————————————————————————————————————
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo1.png", width=200)
# —————————————————————————————————————————————————————————————
# 2) Inject Bootstrap and custom CSS (before UI elements)
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
      /* Dropzone أخضر */
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
      /* زر Browse داخل Dropzone */
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
    </style>
    """,
    unsafe_allow_html=True,
)

# —————————————————————————————————————————————————————————————
# 3) Display title and description
# —————————————————————————————————————————————————————————————
st.markdown(
    """
    <p style="
      text-align: center;
      font-size: 24px;
      font-weight: bold;
      color: #000;
      margin-top: 0.5rem;
    ">
      الرجاء قم بتحميل الملف هنا
    </p>
    """,
    unsafe_allow_html=True,
)

# —————————————————————————————————————————————————————————————
# 4) Load model and related artifacts
# —————————————————————————————————————————————————————————————
artifacts = joblib.load("final_model.pkl")
model                 = artifacts["model"]
features              = artifacts["features"]
danger_map            = artifacts["danger_map"]
weights               = artifacts["weights"]
group1_locations      = artifacts["group1_locations"]
group2_locations      = artifacts["group2_locations"]
group3_locations      = artifacts["group3_locations"]
group1_classification = artifacts["group1_classification"]
group2_classification = artifacts["group2_classification"]
group3_classification = artifacts["group3_classification"]

# —————————————————————————————————————————————————————————————
# 5) Helper functions
# —————————————————————————————————————————————————————————————
# Classify site based on location groups

def classify_site_attribute(location):
    if location in group1_locations:
        return group1_classification
    elif location in group2_locations:
        return group2_classification
    elif location in group3_locations:
        return group3_classification
    else:
        return "غير محدد"
# Mapping classifications to numeric ranks

site_rank_map = {
    group2_classification: 3,
    group1_classification: 2,
    group3_classification: 1,
    "غير محدد": 0
}
# Apply contextual boost based on custom logic

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
# 6) Upload file and stop if not uploaded
# —————————————————————————————————————————————————————————————
uploaded_file = st.file_uploader(
    label="", 
    type=["xlsx"]
)
if not uploaded_file:
    st.info("لم يتم رفع الملف بعد")
    st.stop()


# —————————————————————————————————————————————————————————————
# 7) Read and preprocess input data
# —————————————————————————————————————————————————————————————
df = pd.read_excel(uploaded_file)
if "عدد تكرار البلاغ" in df.columns:
    df.rename(columns={"عدد تكرار البلاغ": "عدد البلاغات"}, inplace=True)

df["درجة الخطورة"] = df["نوع الخدمة"].map(danger_map)
df["صفة الموقع"]  = df["موقع البلاغ"].apply(classify_site_attribute)
df["صفة الموقع"]  = df["صفة الموقع"].map(site_rank_map)
# —————————————————————————————————————————————————————————————
# 8) Calculate score and boost
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
# 9) Predict priority and show top 10 results in a Bootstrap-styled table
# —————————————————————————————————————————————————————————————
X_new = df[features]
df["مستوى_الأولوية"] = model.predict(X_new)

st.success("!تم التقييم")

html_table = (
    df.head(10)[[ # Display only top 10 rows
        "نوع الخدمة","موقع البلاغ","عدد السكان",
        "عدد البلاغات","درجة الخطورة","صفة الموقع",
        "مستوى_الأولوية"
    ]]
    .to_html(index=False, classes="table table-dark table-striped")
)
st.markdown(html_table, unsafe_allow_html=True)

# —————————————————————————————————————————————————————————————
# 10) Download button for results as CSV
# —————————————————————————————————————————————————————————————
csv_data = df.to_csv(index=False)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.download_button(
        label=" CSV تحميل النتائج",
        data=csv_data,
        file_name="نتائج_الأولوية.csv",
        mime="text/csv"
    )

