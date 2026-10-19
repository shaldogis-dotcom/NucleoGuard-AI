
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
st.write("Current denoising strength:", sigma)
st.write("Image difference:", float(np.mean(np.abs(noisy - enhanced))))
change_map = np.abs(noisy - enhanced)

# تحديد أعلى مستوى للتغير لعرض الخريطة بوضوح
vmax = max(float(np.percentile(change_map, 99)), 1e-6)

# تحديد المناطق ذات التغير الأعلى
threshold = float(np.percentile(change_map, 95))
high_change = change_map >= threshold

st.header("ROI Analysis")

st.write(
    "Select a square region of interest (ROI) "
    "to compare image intensity before and after denoising."
)

height, width = noisy.shape

col1, col2 = st.columns(2)

with col1:
    roi_x = st.slider(
        "ROI horizontal position",
        min_value=0,
        max_value=max(0, width - 1),
        value=width // 2
    )

with col2:
    roi_y = st.slider(
        "ROI vertical position",
        min_value=0,
        max_value=max(0, height - 1),
        value=height // 2
    )

roi_size = min(20, height, width)

x_start = min(max(0, roi_x - roi_size // 2), width - roi_size)
y_start = min(max(0, roi_y - roi_size // 2), height - roi_size)

roi_before = noisy[
    y_start:y_start + roi_size,
    x_start:x_start + roi_size
]

roi_after = enhanced[
    y_start:y_start + roi_size,
    x_start:x_start + roi_size
]

before_mean = float(np.mean(roi_before))
after_mean = float(np.mean(roi_after))

col1, col2, col3 = st.columns(3)

col1.metric("ROI size", f"{roi_size} × {roi_size}")
col2.metric("Mean before", f"{before_mean:.4f}")
col3.metric(
    "Mean after",
    f"{after_mean:.4f}",
    delta=f"{after_mean - before_mean:+.4f}"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[0].add_patch(
    plt.Rectangle(
        (x_start, y_start),
        roi_size,
        roi_size,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
)
axes[0].set_title("ROI before denoising")

axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
axes[1].add_patch(
    plt.Rectangle(
        (x_start, y_start),
        roi_size,
        roi_size,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
)
axes[1].set_title("ROI after denoising")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "ROI mean intensity is an experimental comparison only. "
    "It is not an SUV measurement or a validated diagnostic metric."
)

st.subheader("ROI Preservation Review")

roi_change = abs(after_mean - before_mean)


st.write("ROI change:", f"{roi_change_percent:.4f}%")
st.write("Selected threshold:", review_limit)

if review_limit < 50:
    st.warning("TEST: Review alert is active.")
else:
    st.info("TEST: Review alert is inactive.")


st.metric(
    "Absolute ROI mean change",
    f"{roi_change:.4f}"
)

st.metric(
    "Relative ROI mean change",
    f"{roi_change_percent:.2f}%"
)

review_limit = st.slider(
    "ROI change review threshold (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)


st.write("Debug — ROI change:", f"{roi_change_percent:.4f}%")
st.write("Debug — Review threshold:", f"{review_limit}%")

if roi_change_percent >= review_limit:
    st.warning(
        "Review recommended: ROI mean intensity change "
        "meets or exceeds the selected threshold."
    )
else:
    st.info(
        "ROI mean intensity change is below the selected threshold."
    )

st.caption(
    "Experimental screening indicator only. "
    "The threshold is user-defined and is not clinically validated."
)

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
