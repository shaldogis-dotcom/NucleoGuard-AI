
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import io

st.set_page_config(page_title="NucleoGuard AI", page_icon="🛡️",
                   layout="wide")

st.title("🛡️ NucleoGuard AI")
st.subheader("Nuclear Medicine Image Quality Assistant")
st.warning("Research prototype only — not validated for clinical diagnosis.")

def create_demo_image(size=256, seed=42):
    y, x = np.mgrid[:size, :size]
    image = 0.12 * np.exp(
        -((x-128)**2 + (y-128)**2) / (2*75**2))
    image += 0.65 * np.exp(
        -((x-155)**2 + (y-110)**2) / (2*13**2))
    image += 0.30 * np.exp(
        -((x-85)**2 + (y-160)**2) / (2*9**2))
    image /= image.max()
    rng = np.random.default_rng(seed)
    noisy = np.clip(image + rng.normal(0, 0.08, image.shape), 0, 1)
    return image, noisy

def load_image(upload):
    arr = np.asarray(Image.open(upload).convert("L"), dtype=float)
    return arr / 255.0

def calculate_metrics(ref, candidate):
    mae = np.mean(np.abs(ref - candidate))
    mse = np.mean((ref - candidate)**2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    return mae, mse, psnr

st.sidebar.header("Settings")
source = st.sidebar.radio("Image source", ["Demo image", "Upload image"])
sigma = st.sidebar.slider("Denoising strength", 0.3, 3.0, 1.2, 0.1)

reference = None

if source == "Demo image":
    reference, noisy = create_demo_image()
else:
    uploaded = st.file_uploader(
        "Upload a grayscale image",
        type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()
    noisy = load_image(uploaded)

    ref_file = st.file_uploader(
        "Optional: upload matching reference image",
        type=["png", "jpg", "jpeg", "tif", "tiff"])
    if ref_file is not None:
        reference = load_image(ref_file)
        if reference.shape != noisy.shape:
            st.error("Reference and test images must have the same dimensions.")
            st.stop()

enhanced = gaussian_filter(noisy, sigma=sigma)

change_map = np.abs(noisy - enhanced)

# تحديد أعلى مستوى للتغير لعرض الخريطة بوضوح
vmax = max(float(np.percentile(change_map, 99)), 1e-6)

# تحديد المناطق ذات التغير الأعلى
threshold = float(np.percentile(change_map, 95))
high_change = change_map >= threshold

st.header("Image comparison and change detection")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Before denoising")

axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("After denoising")

heatmap = axes[2].imshow(
    change_map,
    cmap="hot",
    vmin=0,
    vmax=vmax
)
axes[2].contour(
    high_change.astype(float),
    levels=[0.5],
    colors="cyan",
    linewidths=0.7
)
axes[2].set_title("Change map — top 5%")
fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04,
             label="Absolute intensity change")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Cyan outlines mark pixels in the highest 5% of observed changes. "
    "This is a relative change indicator, not a calibrated uncertainty map."
)

st.subheader("Change detection alert")

change_limit = st.slider(
    "Minimum intensity change",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01
)

changed_pixels = change_map > change_limit

change_percentage = (
    np.count_nonzero(changed_pixels)
    / changed_pixels.size
) * 100

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Pixels with significant change",
        f"{change_percentage:.2f}%"
    )

with col2:
    st.metric(
        "Change threshold",
        f"{change_limit:.2f}"
    )

if change_percentage >= 10:
    st.warning(
        "Review recommended: a considerable portion "
        "of the image changed during enhancement."
    )
else:
    st.success(
        "No large-scale change detected at the selected threshold. "
        "This does not guarantee preservation of all diagnostic details."
    )

st.caption(
    "This is an experimental image-change indicator, "
    "not a validated measure of diagnostic accuracy or uncertainty."
)
