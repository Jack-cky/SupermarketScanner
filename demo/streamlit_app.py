import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supermarketscanner import SupermarketScanner


torch.classes.__path__ = []


def st_setup_page() -> None:
    st.set_page_config(
        page_title="SupermarketScanner",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


@st.cache_resource
def st_load_model() -> None:

    return SupermarketScanner(os.path.join("models", "fine_tune.onnx"))


def st_describe_demo() -> None:
    st.html(
        """
        <style>
            h1 {
                color: rgb(158,47,129);
                font-size: 56px;
                margin: 0;
                text-align: center;
            }
            h2 {
                font-style: italic;
                margin-top: -15px;
                text-align: center;
            }
            a {
                color: inherit;
            }
        </style>
        <h1>SupermarketScanner</h1>
        <h2>Your business solution to prevent shoplifting</h2>
        <p>SupermarketScanner is an AI-driven system designed to enhance the self-checkout experience. It simplifies the process for customers, saving time, while helping retailers prevent theft by addressing situations where customers <a href="https://www.thestandard.com.hk/breaking-news/section/4/202373/Two-top-doctors-each-fined-HK$5,000-for-stealing-food-from-AEON-in-Whampoa" target="_blank">forget to scan items or claim distractions</a>.</p>
        """
    )

def st_get_basket() -> None:
    img_custom = st.file_uploader(
        "Upload your basket here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if img_custom is None:
        img = image_select(
            "Sample transactions",
            [
                os.path.join("demo", "static", "breakfast_set.jpg"),
                os.path.join("demo", "static", "lunch_set.jpg"),
                os.path.join("demo", "static", "beverage_bulk_buyer.jpg"),
            ],
            captions=["Breakfast Set", "Lunch Set", "Beverage Bulk Buyer"],
            index=0,
        )
    else:
        img = Image.open(img_custom)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


def st_scan_basket(model, img) -> None:
    st.subheader(
        "Screen on Self-Checkout Kiosk",
        divider="rainbow",
    )

    img_pred, _ = model.scan(
        src=img,
        summary=False,
        save=False,
    )

    st.image(img_pred, channels="BGR")


def st_show_demo() -> None:
    model = st_load_model()

    div_left, div_right = st.columns([2, 1])
    with div_right:
        img = st_get_basket()
    with div_left:
        st_scan_basket(model, img)


def main():
    st_setup_page()
    st_describe_demo()
    st_show_demo()


if __name__ == "__main__":
    main()
