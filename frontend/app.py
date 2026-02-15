"""
Streamlit Frontend
===================
Interactive UI for the Logistics Delay Prediction API.

Run:
    streamlit run frontend/app.py
"""

import os
import time
from urllib.parse import urlparse

import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────
def normalize_api_url(raw_api_url: str) -> str:
    """Normalize API URL across local and Render deployments."""
    candidate = (raw_api_url or "").strip().rstrip("/")
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
if API_URL and not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

    if not candidate:
        return "http://localhost:8000"

    if not candidate.startswith(("http://", "https://")):
        # Render can sometimes expose short host values like `logistics-api-xxxx`.
        if "." not in candidate and candidate not in {"localhost", "127.0.0.1"}:
            candidate = f"{candidate}.onrender.com"
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.hostname and "." not in parsed.hostname:
        if parsed.hostname not in {"localhost", "127.0.0.1"}:
            host = f"{parsed.hostname}.onrender.com"
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or ""
            candidate = f"{parsed.scheme}://{host}{port}{path}"

    return candidate


RAW_API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_URL = normalize_api_url(RAW_API_URL)

st.set_page_config(
    page_title="Logistics Delay Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.4rem 0 0; opacity: 0.8; font-size: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        color: white;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.82rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.5px; }

    .result-early  { border-left: 4px solid #00d2ff; }
    .result-ontime { border-left: 4px solid #00e676; }
    .result-late   { border-left: 4px solid #ff5252; }

    .impact-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .impact-box h4 { margin: 0 0 0.5rem; font-weight: 600; }
    .impact-box .amount { font-size: 1.8rem; font-weight: 700; color: #ff9800; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    div[data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }

    .history-table th { background: #1a1a2e; color: white; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📦 Logistics Delay Predictor</h1>
    <p>Predict whether a shipment will arrive <b>Early</b>, <b>On-time</b>, or <b>Late</b> — powered by ML with SHAP explainability</p>
</div>
""", unsafe_allow_html=True)


# ── API health check ────────────────────────────────────────────
def check_api():
    """Check API health with retries and return status details."""
    max_attempts = 12  # free tier cold starts can take ~1 min
    for i in range(max_attempts):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code != 200:
                raise RuntimeError(f"Health endpoint returned {r.status_code}")

            payload = r.json()
            if payload.get("model_loaded", False):
                return True, None

            # API is reachable but model is still warming up.
            if i < max_attempts - 1:
                with st.spinner(f"API reachable, model loading... (Attempt {i+1}/{max_attempts})"):
                    time.sleep(5)
                continue
            return False, "API is reachable, but model is still loading. Check backend logs and MODEL_PATH."

        except Exception as exc:
            if i < max_attempts - 1:
                with st.spinner(f"Waking up API... (Attempt {i+1}/{max_attempts})"):
                    time.sleep(5)
                continue
            return False, str(exc)

    return False, "Unknown API startup issue"


api_live, api_error = check_api()
if not api_live:
    normalization_note = ""
    raw_value = (RAW_API_URL or "").strip()
    if raw_value and raw_value.rstrip("/") != API_URL:
        normalization_note = f"Normalized from `API_URL={raw_value}` to `{API_URL}`.\n\n"

    st.error(
        "⚠️ **API not ready.**\n\n"
        f"Configured `API_URL`: `{API_URL}`\n\n"
        f"{normalization_note}"
        f"Details: `{api_error}`\n\n"
        "If you are deploying on Render, confirm the frontend `API_URL` env var points to your API service URL "
        "(for example `https://logistics-api-xxxx.onrender.com`)."
        f"Details: `{api_error}`\n\n"
        "If you are deploying on Render, confirm the frontend `API_URL` env var points to your API service URL."
    )
    st.stop()


# ── Sidebar: Input Form ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Shipment Details")
    st.markdown("---")

    # ── Order & Financial ────────────────────────────────────────
    st.markdown("**💰 Financial**")
    profit_per_order = st.number_input("Profit per order ($)", value=50.0, step=5.0)
    sales_per_customer = st.number_input("Sales per customer ($)", value=200.0, step=10.0)
    sales = st.number_input("Sale amount ($)", value=200.0, step=10.0)
    product_price = st.number_input("Product price ($)", value=100.0, step=5.0)
    order_item_product_price = st.number_input("Item price ($)", value=100.0, step=5.0)
    order_item_quantity = st.number_input("Quantity", value=2, min_value=1, step=1)

    st.markdown("---")
    st.markdown("**🏷️ Discounts & Profit**")
    order_item_discount = st.number_input("Discount ($)", value=10.0, step=1.0)
    order_item_discount_rate = st.slider("Discount rate", 0.0, 1.0, 0.05, 0.01)
    order_item_profit_ratio = st.slider("Profit ratio", -1.0, 1.0, 0.25, 0.01)
    discount_ratio = st.number_input("Discount / Price ratio", value=0.1, step=0.01, format="%.3f")
    profit_margin = st.number_input("Profit margin", value=0.25, step=0.01, format="%.3f")

    st.markdown("---")
    st.markdown("**📍 Location**")
    latitude = st.number_input("Latitude", value=28.6, step=0.1, format="%.4f")
    longitude = st.number_input("Longitude", value=77.2, step=0.1, format="%.4f")

    st.markdown("---")
    st.markdown("**📅 Date Features**")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        order_dayofweek = st.selectbox("Order day", list(range(7)),
                                        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                                        index=3)
        order_month = st.selectbox("Order month", list(range(1, 13)), index=5)
        order_quarter = st.selectbox("Order quarter", [1, 2, 3, 4], index=1)
    with col_d2:
        shipping_dayofweek = st.selectbox("Ship day", list(range(7)),
                                           format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                                           index=5)
        shipping_month = st.selectbox("Ship month", list(range(1, 13)), index=5)
        shipping_lead_days = st.number_input("Lead days", value=3.5, step=0.5, min_value=0.0)

    st.markdown("---")
    st.markdown("**📋 Categories**")
    payment_type = st.selectbox("Payment", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
    category_name = st.text_input("Category", value="Cleats")
    customer_country = st.text_input("Country", value="EE. UU.")
    customer_segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    department_name = st.text_input("Department", value="Fan Shop")
    market = st.selectbox("Market", ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"])
    order_region = st.text_input("Region", value="Central America")
    order_status = st.selectbox("Order status", [
        "COMPLETE", "CLOSED", "PENDING", "PENDING_PAYMENT",
        "SUSPECTED_FRAUD", "CANCELED", "ON_HOLD",
    ])
    shipping_mode = st.selectbox("Shipping mode", [
        "Standard Class", "Second Class", "First Class", "Same Day",
    ])

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Delay", use_container_width=True, type="primary")


# ── Prediction Logic ────────────────────────────────────────────
if predict_btn:
    payload = {
        "profit_per_order": profit_per_order,
        "sales_per_customer": sales_per_customer,
        "latitude": latitude,
        "longitude": longitude,
        "order_item_discount": order_item_discount,
        "order_item_discount_rate": order_item_discount_rate,
        "order_item_product_price": order_item_product_price,
        "order_item_profit_ratio": order_item_profit_ratio,
        "order_item_quantity": order_item_quantity,
        "sales": sales,
        "product_price": product_price,
        "order_dayofweek": order_dayofweek,
        "order_month": order_month,
        "order_quarter": order_quarter,
        "shipping_dayofweek": shipping_dayofweek,
        "shipping_month": shipping_month,
        "shipping_lead_days": shipping_lead_days,
        "discount_ratio": discount_ratio,
        "profit_margin": profit_margin,
        "payment_type": payment_type,
        "category_name": category_name,
        "customer_country": customer_country,
        "customer_segment": customer_segment,
        "department_name": department_name,
        "market": market,
        "order_region": order_region,
        "order_status": order_status,
        "shipping_mode": shipping_mode,
    }

    with st.spinner("Predicting..."):
        start = time.time()
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            latency = time.time() - start
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.stop()

    # ── Result Display ───────────────────────────────────────────
    class_name = result["class_name"]
    label = result["label"]
    confidence = result["confidence"]
    max_conf = result["max_confidence"]
    top_features = result.get("top_features", [])

    color_map = {"Early": "#00d2ff", "On-time": "#00e676", "Late": "#ff5252"}
    emoji_map = {"Early": "🟢", "On-time": "🔵", "Late": "🔴"}
    css_class = {"Early": "result-early", "On-time": "result-ontime", "Late": "result-late"}

    color = color_map.get(class_name, "#aaa")
    emoji = emoji_map.get(class_name, "⚪")

    # ── Metric cards row ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card {css_class.get(class_name, '')}">
            <div class="label">Prediction</div>
            <div class="value">{emoji} {class_name}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Confidence</div>
            <div class="value">{max_conf:.0%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Latency</div>
            <div class="value">{latency:.2f}s</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Label</div>
            <div class="value">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: Confidence Gauge + SHAP ───────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📊 Class Probabilities")
        # Donut / bar chart of class probabilities
        labels_list = list(confidence.keys())
        values_list = list(confidence.values())
        colors_list = [color_map.get(l, "#888") for l in labels_list]

        fig_conf = go.Figure(go.Bar(
            x=values_list,
            y=labels_list,
            orientation="h",
            marker=dict(color=colors_list, line=dict(width=0)),
            text=[f"{v:.1%}" for v in values_list],
            textposition="inside",
            textfont=dict(color="white", size=14, family="Inter"),
        ))
        fig_conf.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=14, color="white")),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_right:
        st.markdown("#### 🔍 Top Feature Impacts (SHAP)")
        if top_features:
            feat_names = [f["feature"].replace("num__", "").replace("cat__", "") for f in top_features]
            feat_values = [f["impact"] for f in top_features]
            feat_colors = ["#00e676" if v >= 0 else "#ff5252" for v in feat_values]

            fig_shap = go.Figure(go.Bar(
                x=feat_values,
                y=feat_names,
                orientation="h",
                marker=dict(color=feat_colors, line=dict(width=0)),
                text=[f"{v:+.4f}" for v in feat_values],
                textposition="outside",
                textfont=dict(color="white", size=11, family="Inter"),
            ))
            fig_shap.update_layout(
                height=250,
                margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickfont=dict(color="rgba(255,255,255,0.5)")),
                yaxis=dict(showgrid=False, autorange="reversed",
                           tickfont=dict(size=12, color="white")),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP values not available for this prediction.")

    # ── Business Impact Calculator ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💼 Business Impact Estimate")

    order_value = sales
    late_penalty_pct = 0.15  # assume 15% cost penalty for late deliveries
    early_savings_pct = 0.05  # 5% saved by early prediction

    bi_c1, bi_c2, bi_c3 = st.columns(3)
    if class_name == "Late":
        penalty = order_value * late_penalty_pct
        with bi_c1:
            st.markdown(f"""
            <div class="impact-box">
                <h4>⚠️ Potential Loss</h4>
                <div class="amount">₹{penalty:,.0f}</div>
                <p style="opacity:0.65;margin-top:0.3rem">~{late_penalty_pct:.0%} of order value if delayed</p>
            </div>""", unsafe_allow_html=True)
        with bi_c2:
            st.markdown(f"""
            <div class="impact-box">
                <h4>💡 Recommendation</h4>
                <p>Escalate to logistics team. Consider expedited shipping or route optimization to avoid delay.</p>
            </div>""", unsafe_allow_html=True)
        with bi_c3:
            st.markdown(f"""
            <div class="impact-box">
                <h4>🎯 Risk Level</h4>
                <div class="amount" style="color:#ff5252">HIGH</div>
            </div>""", unsafe_allow_html=True)
    elif class_name == "Early":
        savings = order_value * early_savings_pct
        with bi_c1:
            st.markdown(f"""
            <div class="impact-box">
                <h4>✅ Estimated Savings</h4>
                <div class="amount" style="color:#00e676">₹{savings:,.0f}</div>
                <p style="opacity:0.65;margin-top:0.3rem">Early delivery improves customer satisfaction</p>
            </div>""", unsafe_allow_html=True)
        with bi_c2:
            st.markdown(f"""
            <div class="impact-box">
                <h4>💡 Recommendation</h4>
                <p>No action needed. Shipment is on track for early delivery. Consider notifying the customer.</p>
            </div>""", unsafe_allow_html=True)
        with bi_c3:
            st.markdown(f"""
            <div class="impact-box">
                <h4>🎯 Risk Level</h4>
                <div class="amount" style="color:#00e676">LOW</div>
            </div>""", unsafe_allow_html=True)
    else:
        with bi_c1:
            st.markdown(f"""
            <div class="impact-box">
                <h4>📦 Status</h4>
                <div class="amount" style="color:#00d2ff">ON TRACK</div>
                <p style="opacity:0.65;margin-top:0.3rem">Shipment expected on time</p>
            </div>""", unsafe_allow_html=True)
        with bi_c2:
            st.markdown(f"""
            <div class="impact-box">
                <h4>💡 Recommendation</h4>
                <p>Standard monitoring. No intervention required.</p>
            </div>""", unsafe_allow_html=True)
        with bi_c3:
            st.markdown(f"""
            <div class="impact-box">
                <h4>🎯 Risk Level</h4>
                <div class="amount" style="color:#00d2ff">NORMAL</div>
            </div>""", unsafe_allow_html=True)

    # ── Save to history ──────────────────────────────────────────
    st.session_state.history.append({
        "Time": time.strftime("%H:%M:%S"),
        "Prediction": f"{emoji} {class_name}",
        "Confidence": f"{max_conf:.0%}",
        "Latency": f"{latency:.2f}s",
        "Shipping Mode": shipping_mode,
        "Market": market,
        "Order Value ($)": f"{sales:.0f}",
    })


# ── Prediction History ──────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📜 Prediction History")
    st.dataframe(
        list(reversed(st.session_state.history)),
        use_container_width=True,
        hide_index=True,
    )
