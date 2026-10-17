
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

st.header("Image comparison")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Before")
axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("After")
axes[2].imshow(change_map, cmap="hot")
axes[2].set_title("Change map")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.header("Quality metrics")

if reference is not None:
    mae1, mse1, psnr1 = calculate_metrics(reference, noisy)
    mae2, mse2, psnr2 = calculate_metrics(reference, enhanced)

    st.table({
        "Metric": ["MAE", "MSE", "PSNR (dB)"],
        "Before": [round(mae1, 5), round(mse1, 5), round(psnr1, 2)],
        "After": [round(mae2, 5), round(mse2, 5), round(psnr2, 2)]
    })
else:
    st.info("Upload a matching reference image to calculate MAE, MSE and PSNR.")

st.header("ROI measurement")
h, w = noisy.shape
cx = st.slider("ROI center X", 0, w-1, w//2)
cy = st.slider("ROI center Y", 0, h-1, h//2)
radius = st.slider("ROI radius", 3, max(3, min(h, w)//4), 12)

yy, xx = np.ogrid[:h, :w]
roi = (xx-cx)**2 + (yy-cy)**2 <= radius**2
before = float(noisy[roi].mean())
after = float(enhanced[roi].mean())
change = 100 * abs(after-before) / max(abs(before), 1e-12)

c1, c2, c3 = st.columns(3)
c1.metric("Mean before", f"{before:.4f}")
c2.metric("Mean after", f"{after:.4f}")
c3.metric("Relative change", f"{change:.2f}%")

limit = st.sidebar.slider("Demo alert threshold (%)", 1.0, 20.0, 5.0)
if change > limit:
    st.warning("Review suggested: ROI measurement changed above demo threshold.")
else:
    st.info("No alert under the selected demo threshold.")

buffer = io.BytesIO()
plt.imsave(buffer, enhanced, cmap="gray", vmin=0, vmax=1, format="png")
st.download_button(
    "Download enhanced image",
    data=buffer.getvalue(),
    file_name="nucleoguard_enhanced.png",
    mime="image/png"
)

st.caption(
    "Experimental prototype only. Change maps are not calibrated uncertainty "
    "maps. ROI intensity is not clinical SUV. Not for diagnosis or patient care."
)
