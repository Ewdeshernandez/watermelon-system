import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Bode Plot")

uploaded_files = st.file_uploader("Upload Bode CSV", accept_multiple_files=True)

if not uploaded_files:
    st.info("Carga CSV de Bode")
    st.stop()

df_list = []

for file in uploaded_files:
    df = pd.read_csv(file)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# =========================
# LIMPIEZA BODE
# =========================
df = df.rename(columns=lambda x: x.strip())

df["rpm"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")

df = df.dropna(subset=["rpm", "amp", "phase"])
df = df.sort_values("rpm")

# =========================
# CONTROL EJE X
# =========================
st.sidebar.markdown("### X Axis Control")

auto_scale = st.sidebar.checkbox("Auto Scale", value=True)

if not auto_scale:
    min_rpm = st.sidebar.number_input("Min RPM", value=int(df["rpm"].min()))
    max_rpm = st.sidebar.number_input("Max RPM", value=int(df["rpm"].max()))
else:
    min_rpm = df["rpm"].min()
    max_rpm = df["rpm"].max()

# =========================
# CURSORES (RPM)
# =========================
st.sidebar.markdown("### Cursors")

cursor_A = st.sidebar.slider("Cursor A (RPM)", int(min_rpm), int(max_rpm), int(df["rpm"].iloc[0]))
cursor_B = st.sidebar.slider("Cursor B (RPM)", int(min_rpm), int(max_rpm), int(df["rpm"].iloc[-1]))

def get_closest(df, rpm):
    return df.iloc[(df["rpm"] - rpm).abs().argsort()[:1]]

row_A = get_closest(df, cursor_A).iloc[0]
row_B = get_closest(df, cursor_B).iloc[0]

# =========================
# FIGURA
# =========================
fig = go.Figure()

# AMPLITUD (LÍNEA FINA)
fig.add_trace(go.Scatter(
    x=df["rpm"],
    y=df["amp"],
    name="Amplitude",
    line=dict(width=2)  # 🔥 más fino
))

# FASE
fig.add_trace(go.Scatter(
    x=df["rpm"],
    y=df["phase"],
    name="Phase",
    yaxis="y2",
    line=dict(width=2, dash="dot")
))

# CURSORES
fig.add_vline(x=row_A["rpm"], line_dash="dash", line_color="orange")
fig.add_vline(x=row_B["rpm"], line_dash="dash", line_color="green")

# =========================
# LAYOUT
# =========================
fig.update_layout(
    xaxis=dict(
        title="Speed (RPM)",
        range=[min_rpm, max_rpm]
    ),
    yaxis=dict(
        title="Amplitude (mil pp)"
    ),
    yaxis2=dict(
        title="Phase (°)",
        overlaying="y",
        side="right"
    ),
    template="plotly_white",
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# PANEL INFO (TIPO SYSTEM1)
# =========================
st.markdown("### Bode Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Cursor A**")
    st.write(f"RPM: {int(row_A['rpm'])}")
    st.write(f"Amplitude: {row_A['amp']:.3f}")
    st.write(f"Phase: {row_A['phase']:.2f}")

with col2:
    st.markdown("**Cursor B**")
    st.write(f"RPM: {int(row_B['rpm'])}")
    st.write(f"Amplitude: {row_B['amp']:.3f}")
    st.write(f"Phase: {row_B['phase']:.2f}")

