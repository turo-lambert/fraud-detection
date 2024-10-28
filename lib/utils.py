"""Utils for the project."""
import streamlit as st
from PIL import Image


@st.cache_data
def load_image_w_cache(local_path: str) -> Image.Image:
    """Loads an image from the given local file path with Streamlit caching.

    Parameters:
    local_path (str): The local file path of the image.

    Returns:
    Image: The loaded image object.
    """
    return Image.open(local_path)
