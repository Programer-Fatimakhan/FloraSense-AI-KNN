
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

st.set_page_config(
    page_title="FloraSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained files
@st.cache_resource
def load_model():
    model = joblib.load("models/knn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    names = joblib.load("models/flower_names.pkl")
    results = joblib.load("models/results.pkl")
    return model, scaler, names, results

try:
    model, scaler, flower_names, results = load_model()
except Exception as e:
    st.error(f"Model files could not be loaded: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# PREMIUM DARK THEME
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071b16, #102b22);
    color: #e6f2e9;
}

header[data-testid="stHeader"] {
    background: #071b16;
}

section[data-testid="stSidebar"] {
    background: #0b211a;
    border-right: 1px solid #244d3b;
}

section[data-testid="stSidebar"] * {
    color: #e6f2e9 !important;
}

h1, h2, h3 {
    color: #b9edc7 !important;
}

p, label, span, li {
    color: #e1eee5;
}

div[data-testid="stMetric"] {
    background: #15382b;
    border: 1px solid #376b50;
    padding: 20px;
    border-radius: 15px;
}

div[data-testid="stMetric"] label {
    color: #a9d7b6 !important;
}

div[data-testid="stMetricValue"] {
    color: #d4f5a5 !important;
}

.stButton button {
    background: linear-gradient(90deg, #18764d, #269b65);
    color: white !important;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    background: #36b77a;
    color: white !important;
    border: 1px solid #b9edc7;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #102b22;
    border-radius: 15px;
}

input, textarea {
    background-color: #15382b !important;
    color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: #15382b;
    color: white;
}

hr {
    border-color: #376b50;
}

.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("# 🌿 FloraSense AI")
st.sidebar.caption("SMART FLOWER INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🌸 Predict Flower",
        "📊 Model Analytics",
        "📋 Prediction History"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**AI MODEL INFORMATION**")
st.sidebar.caption("Algorithm: K-Nearest Neighbors")
st.sidebar.caption("Dataset: Iris Flower Dataset")
st.sidebar.caption("Framework: Scikit-learn")

# DASHBOARD
if page == "🏠 Dashboard":

    st.title("🌿 FloraSense AI")
    st.subheader("Intelligent Flower Classification System")

    st.write(
        "An interactive Machine Learning application powered by "
        "the K-Nearest Neighbors algorithm."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MODEL ACCURACY",
        f"{results['accuracy'] * 100:.2f}%"
    )

    c2.metric("ALGORITHM", "KNN")

    c3.metric("FLOWER SPECIES", len(flower_names))

    st.divider()

    st.subheader("🌺 Explore FloraSense")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Smart Prediction")
        st.write(
            "Enter flower measurements and let our trained "
            "Machine Learning model identify the species."
        )

    with col2:
        st.markdown("### Model Intelligence")
        st.write(
            "Explore model accuracy, confusion matrix and "
            "classification performance."
        )

    st.divider()

    st.subheader("🌱 Flower Categories")

    flower_cols = st.columns(3)

    descriptions = {
        "setosa": "Distinctive small petals and sepals.",
        "versicolor": "Medium-sized petals and balanced features.",
        "virginica": "Generally larger petals and sepals."
    }

    for i, name in enumerate(flower_names):
        with flower_cols[i]:
            st.markdown(f"### 🌸 {name.title()}")
            st.write(descriptions.get(name, "Iris flower species."))

# PREDICTION
elif page == "🌸 Predict Flower":

    st.title("🌸 Smart Flower Prediction")
    st.write("Enter the four flower measurements below.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)", 0.1, 10.0, 5.1, 0.1
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)", 0.1, 10.0, 3.5, 0.1
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)", 0.1, 10.0, 1.4, 0.1
        )

        petal_width = st.number_input(
            "Petal Width (cm)", 0.1, 10.0, 0.2, 0.1
        )

    st.divider()

    k_value = st.select_slider(
        "Choose K Neighbors",
        options=[3, 5, 7, 9, 11],
        value=5
    )

    if st.button("✨ ANALYZE FLOWER"):

        input_data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        input_scaled = scaler.transform(input_data)

        prediction_model = joblib.load("models/knn_model.pkl")
        prediction_model.set_params(n_neighbors=k_value)

        prediction = prediction_model.predict(input_scaled)[0]
        probabilities = prediction_model.predict_proba(input_scaled)[0]

        flower = flower_names[prediction]
        confidence = probabilities[prediction] * 100

        st.divider()

        st.success(f"🌺 Identified Species: {flower.title()}")

        a, b = st.columns(2)

        a.metric("PREDICTED FLOWER", flower.title())
        b.metric("MODEL CONFIDENCE", f"{confidence:.2f}%")

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Flower": [name.title() for name in flower_names],
            "Probability": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Flower"),
            color="#43c987"
        )

        st.session_state.history.append({
            "Flower": flower.title(),
            "K Value": k_value,
            "Confidence": round(confidence, 2),
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

# ANALYTICS
elif page == "📊 Model Analytics":

    st.title("📊 Model Performance Analytics")
    st.write("Explore the performance of your trained KNN model.")

    st.metric(
        "TEST ACCURACY",
        f"{results['accuracy'] * 100:.2f}%"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    matrix = np.array(results["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#102b22")
    ax.set_facecolor("#102b22")

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=flower_names,
        yticklabels=flower_names,
        ax=ax,
        annot_kws={"color": "#102b22", "size": 13}
    )

    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Classification Report")

    report_df = pd.DataFrame(results["report"]).T

    st.dataframe(
        report_df.style.format("{:.2f}"),
        use_container_width=True
    )

# HISTORY
elif page == "📋 Prediction History":

    st.title("📋 Prediction History")
    st.write("Review your previous flower predictions.")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction History",
            csv,
            "flora_predictions.csv",
            "text/csv"
        )

    else:
        st.info(
            "No predictions available yet. "
            "Visit Predict Flower to start."
        )

st.sidebar.divider()
st.sidebar.caption("FloraSense AI © 2026")
st.sidebar.caption("Powered by Machine Learning")
