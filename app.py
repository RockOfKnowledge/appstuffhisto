pip install -r requirements.txt
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Histogram Distribution Fitter", layout="wide")

# -----------------------------------------
# Helper functions
# -----------------------------------------
def safe_len(x):
    """Return length safely even if x is None."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0

def parse_manual_data(text):
    """Convert manual input to numpy array safely."""
    try:
        cleaned = [x.strip() for x in text.replace(",", " ").split()]
        nums = [float(x) for x in cleaned if x != ""]
        return np.array(nums)
    except:
        return np.array([])

def fit_distribution(dist, data):
    return dist.fit(data)

def compute_errors(data, pdf_vals):
    hist_vals, bin_edges = np.histogram(data, bins="auto", density=True)
    if safe_len(hist_vals) == 0:
        return np.nan, np.nan
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if safe_len(data) > 0:
        x_min, x_max = np.min(data), np.max(data)
    else:
        x_min, x_max = 0, 1

    pdf_interp = np.interp(
        bin_centers,
        np.linspace(x_min, x_max, safe_len(pdf_vals)),
        pdf_vals
    )
    mae = np.mean(np.abs(hist_vals - pdf_interp))
    rmse = np.sqrt(np.mean((hist_vals - pdf_interp)**2))
    return mae, rmse

# -----------------------------------------
# Initialize session state
# -----------------------------------------
if "data" not in st.session_state or st.session_state["data"] is None:
    st.session_state["data"] = np.array([])

# -----------------------------------------
# Distributions
# -----------------------------------------
DIST_OPTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma": stats.gamma,
    "Weibull (weibull_min)": stats.weibull_min,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Chi-Square": stats.chi2,
    "Beta": stats.beta,
    "Student t": stats.t,
    "Gumbel": stats.gumbel_r,
    "Cauchy": stats.cauchy,
    "Rayleigh": stats.rayleigh
}

# -----------------------------------------
# Sidebar Input
# -----------------------------------------
with st.sidebar:
    st.title("📊 Data Input")
    manual_text = st.text_area("Enter data (spaces or commas)", height=150)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# -----------------------------------------
# CSV Processing (safe)
# -----------------------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df is None or df.empty:
            st.error("CSV is empty or unreadable.")
            numeric_cols = pd.DataFrame()
        else:
            numeric_cols = df.select_dtypes(include=np.number)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        numeric_cols = pd.DataFrame()

    if numeric_cols.empty:
        st.error("Uploaded CSV has no numeric columns.")
        st.session_state["data"] = np.array([])
    else:
        st.session_state["data"] = numeric_cols.iloc[:, 0].dropna().values

# Fallback to manual input if CSV fails or empty
if safe_len(st.session_state["data"]) == 0:
    st.session_state["data"] = parse_manual_data(manual_text)

data = st.session_state["data"]

# Stop app if no valid data
if safe_len(data) == 0:
    st.warning("Please enter valid numeric data or upload a CSV.")
    st.stop()

# -----------------------------------------
# Main App Layout
# -----------------------------------------
st.title("📈 Histogram Distribution Fitting App")
st.success(f"Dataset loaded with {safe_len(data)} values")
tabs = st.tabs(["🔧 Automatic Fit", "🎛 Manual Fit"])

# =========================================
# Tab 1: Automatic Fit
# =========================================
with tabs[0]:
    st.header("Automatic Fit")
    col1, col2 = st.columns([1, 2])

    with col1:
        dist_name = st.selectbox("Distribution", list(DIST_OPTIONS.keys()))
        dist = DIST_OPTIONS[dist_name]
        params = fit_distribution(dist, data)
        st.subheader("Fitted Parameters:")
        for i, p in enumerate(params):
            st.write(f"param{i}: {p:.4f}")

    with col2:
        # Safe min/max for plotting
        x_min, x_max = (np.min(data), np.max(data)) if safe_len(data) > 0 else (0, 1)
        x = np.linspace(x_min, x_max, 500)
        pdf_vals = dist.pdf(x, *params)
        mae, rmse = compute_errors(data, pdf_vals)
        st.write(f"MAE: {mae:.5f}")
        st.write(f"RMSE: {rmse:.5f}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(data, bins="auto", density=True, alpha=0.5)
        ax.plot(x, pdf_vals, "r-")
        st.pyplot(fig)

# =========================================
# Tab 2: Manual Fit
# =========================================
with tabs[1]:
    st.header("Manual Fit")
    dist_name_m = st.selectbox("Distribution (Manual)", list(DIST_OPTIONS.keys()))
    dist_m = DIST_OPTIONS[dist_name_m]
    init_params = list(dist_m.fit(data))

    sliders = []
    for i, p in enumerate(init_params):
        sliders.append(
            st.slider(
                f"param{i}",
                min_value=float(p * 0.1 if p != 0 else -10),
                max_value=float(p * 3 + 1),
                value=float(p),
                step=0.01
            )
        )

    # Safe min/max for plotting
    x_min, x_max = (np.min(data), np.max(data)) if safe_len(data) > 0 else (0, 1)
    x = np.linspace(x_min, x_max, 500)
    pdf_vals_manual = dist_m.pdf(x, *sliders)
    mae_m, rmse_m = compute_errors(data, pdf_vals_manual)
    st.write(f"MAE: {mae_m:.5f}")
    st.write(f"RMSE: {rmse_m:.5f}")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(data, bins="auto", density=True, alpha=0.5)
    ax2.plot(x, pdf_vals_manual, "g-")
    st.pyplot(fig2)

st.caption("Built with ❤️ using Streamlit & SciPy")


