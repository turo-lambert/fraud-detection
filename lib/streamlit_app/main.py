"""Streamlit app."""
from pathlib import Path

import streamlit as st

from lib.streamlit_app.app_pages import explainability_page, inference_page
from lib.streamlit_app.utils import load_image_w_cache

PAGES = {"Inference": inference_page, "Explainability": explainability_page}
FRISS_LOGO_PATH = Path(__file__).parent / "references" / "friss_logo.png"


def main() -> None:
    """Main function of the app."""
    st.sidebar.image(load_image_w_cache(FRISS_LOGO_PATH), use_column_width=True)
    selection = st.sidebar.selectbox("", list(PAGES.keys()))
    page = PAGES[selection]
    page.write()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
