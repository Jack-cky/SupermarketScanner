import os
from pathlib import Path
import sys

import torch
import cv2
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_image_select import image_select

ROOT_DIR = Path(__file__).resolve().parent.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smktscnr import SupermarketScanner

torch.classes.__path__ = []

MODEL_PATH = ROOT_DIR / "checkpoints" / "smktscnr_ft.onnx"
ASSETS_DIR = ROOT_DIR / "apps" / "demo" / "assets"


def st_setup_page() -> None:
    st.set_page_config(
        page_title="SupermarketScanner",
        page_icon="🛒",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1120px;
                padding-top: 3rem;
                padding-bottom: 2rem;
            }

            .ss-hero-copy {
                color: #475569;
                font-size: 1.02rem;
                line-height: 1.7;
            }

            .ss-preview-label {
                color: #2563EB;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def st_load_model() -> SupermarketScanner:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="Jack-cky/SupermarketScanner",
            filename="checkpoints/smktscnr_ft.onnx",
            revision="main",
            local_dir=str(ROOT_DIR),
            token=os.getenv("HF_TOKEN"),
        )

    return SupermarketScanner(str(MODEL_PATH))


def st_describe_demo() -> None:
    hero_left, hero_right = st.columns([1.7, 0.8], gap="small")

    with hero_left:
        st.title("SupermarketScanner")
        st.subheader(
            "A business solution for preventing shop theft",
            anchor=False,
            divider="rainbow",
        )
        st.markdown(
            (
                '<div class="ss-hero-copy">SupermarketScanner is an '
                "AI-driven system designed to improve the self-checkout "
                "experience. It makes the process quicker and simpler for "
                "customers, while helping retailers tackle theft by "
                "identifying situations where items have not been scanned, "
                "including "
                '<a href="https://www.thestandard.com.hk/breaking-news/'
                'section/4/202373/Two-top-doctors-each-fined-HK$5,000-for-'
                'stealing-food-from-AEON-in-Whampoa" '
                'target="_blank">cases</a> '
                "where customers claim they were distracted or forgot to "
                "scan them.</div>"
            ),
            unsafe_allow_html=True,
        )

    with hero_right:
        st.image(
            str(ASSETS_DIR / "reference.jpg"),
            use_container_width=True,
        )

    st.info(
        (
            "Choose an input source, then review the annotated kiosk preview "
            "on the right."
        ),
        icon=":material/photo_camera:",
    )


def st_get_sample_transactions() -> dict[str, str]:
    return {
        "Breakfast Set": str(ASSETS_DIR / "breakfast_set.jpg"),
        "Lunch Set": str(ASSETS_DIR / "lunch_set.jpg"),
        "Beverage Bulk Buyer": str(ASSETS_DIR / "beverage_bulk_buyer.jpg"),
    }


def st_get_basket() -> tuple[str | np.ndarray | None, str]:
    sample_transactions = st_get_sample_transactions()

    input_mode = st.radio(
        "Input mode",
        options=["Sample transaction", "Upload image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Upload image":
        st.caption(
            "Upload a checkout-counter image from a real or staged "
            "transaction."
        )
        img_custom = st.file_uploader(
            "Basket image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help=(
                "Upload a checkout-counter image to compare against the "
                "bundled sample transactions."
            ),
        )
        if img_custom is None:
            st.info("Upload an image to run the scanner.")
            return None, "Awaiting upload"

        img = Image.open(img_custom).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        st.image(
            img,
            channels="BGR",
            caption="Uploaded basket",
            use_container_width=True,
        )
        return img, "Uploaded basket"

    st.caption(
        "Pick one of the bundled checkout scenes for a quick walkthrough."
    )
    selected_img = image_select(
        "Sample transaction",
        images=list(sample_transactions.values()),
        captions=list(sample_transactions.keys()),
        index=0,
        use_container_width=True,
    )

    selected_label = next(
        (
            label
            for label, path in sample_transactions.items()
            if path == selected_img
        ),
        next(iter(sample_transactions)),
    )
    return sample_transactions[selected_label], selected_label


def st_scan_basket(
    model: SupermarketScanner,
    img: str | np.ndarray,
) -> tuple[np.ndarray | None, dict[str, int]]:
    with st.spinner("Running product detection..."):
        img_pred, qty = model.scan(
            src=img,
            summary=False,
            save=False,
        )

    return img_pred, qty


def st_render_result_panel(
    model: SupermarketScanner,
    img: str | np.ndarray | None,
    basket_label: str,
) -> None:
    with st.container(border=True):
        st.markdown("#### Kiosk preview")
        st.markdown(
            f'<div class="ss-preview-label">{basket_label}</div>',
            unsafe_allow_html=True,
        )
        st.write(
            (
                "Review the annotated frame below to see which products the "
                "model recognised on the checkout counter."
            )
        )

        if img is None:
            st.info(
                "Select a sample transaction or upload an image to generate "
                "the kiosk preview."
            )
            return

        img_pred, qty = st_scan_basket(model, img)
        if img_pred is None:
            st.error(
                "The scan failed for this image. Try another sample or upload "
                "a clearer basket photo."
            )
            return

        st.image(img_pred, channels="BGR", use_container_width=True)
        if not qty:
            st.warning(
                "No items were confidently detected in this basket. Try a "
                "clearer image, a higher-resolution upload, or a bundled "
                "sample."
            )
        else:
            st.caption(
                "Best results come from bright images where products are "
                "visible, front-facing, and minimally occluded."
            )

            st.error(
                "Shoplifting is unlawful behaviour. Deliberately not scanning "
                "items at checkout is a serious offence and should not be "
                "downplayed simply because the person involved holds a "
                "respectable job or enjoys social status."
            )


def st_input_source() -> tuple[str | np.ndarray | None, str]:
    with st.container(border=True):
        st.markdown("#### Input source")
        st.write(
            "Choose how you want to test the scanner. Sample mode is faster. "
            "Upload mode is better for validating your own checkout image."
        )
        return st_get_basket()


def st_show_demo() -> None:
    with st.spinner("Preparing detection model..."):
        model = st_load_model()

    control_col, preview_col = st.columns([0.92, 1.7], gap="small")
    with control_col:
        img, basket_label = st_input_source()

    with preview_col:
        st_render_result_panel(model, img, basket_label)

    st.caption(
        "This PoC demo was inspired by a self-checkout shoplifting "
        "[case](https://www.thestandard.com.hk/breaking-news/section/4/"
        "202373/Two-top-doctors-each-fined-HK$5,000-for-stealing-food-from-"
        "AEON-in-Whampoa). "
        "If someone somehow gets distracted while items keep missing "
        "the scanner, this system is meant to make that excuse harder to sell."
    )


def main() -> None:
    st_setup_page()
    st_describe_demo()
    st_show_demo()


if __name__ == "__main__":
    main()
