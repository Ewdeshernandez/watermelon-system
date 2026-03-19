import streamlit as st


def apply_watermelon_theme() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 0.45rem;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.05), transparent 22%),
                radial-gradient(circle at top right, rgba(14,165,233,0.04), transparent 18%),
                linear-gradient(180deg, #f4f6f8 0%, #eef2f7 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #edf1f6 0%, #e6ebf2 100%);
            border-right: 1px solid #d3dbe5;
        }

        section[data-testid="stSidebar"] * {
            color: #394150;
        }

        .block-container {
            padding-top: 0.65rem;
            padding-bottom: 1.6rem;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            min-height: 48px;
            border-radius: 14px;
            font-weight: 700;
            border: 1px solid #d5dce6;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
            border-color: #b9c6d8;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.78);
            border: 1px solid #d8e0ea;
            border-radius: 22px;
            padding: 10px 12px 4px 12px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.04);
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 12px !important;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div {
            border-radius: 12px !important;
        }

        .wm-surface {
            background: rgba(255,255,255,0.82);
            border: 1px solid #dbe3ec;
            border-radius: 22px;
            padding: 18px 20px;
            box-shadow:
                0 10px 30px rgba(15,23,42,0.04),
                inset 0 1px 0 rgba(255,255,255,0.75);
        }

        .wm-note {
            color: #536174;
            font-size: 0.97rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )