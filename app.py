import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

try:
    import pydicom
    from pydicom.pixels import apply_modality_lut
    DICOM_AVAILABLE = True
except ImportError:
    pydicom = None
    apply_modality_lut = None
    DICOM_AVAILABLE = False


st.set_page_config(
    page_title="NucleoGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Image creation and loading
# -----------------------------
def create_demo_image(size=256, seed=42):
    y, x = np.mgrid[:size, :size]

    image = 0.12 * np.exp(
        -((x - 128) ** 2 + (y - 128) ** 2) / (2 * 75 ** 2)
    )
    image += 0.65 * np.exp(
        -((x - 155) ** 2 + (y - 110) ** 2) / (2 * 13 ** 2)
    )
    image += 0.30 * np.exp(
        -((x - 85) ** 2 + (y - 160) ** 2) / (2 * 9 ** 2)
    )

    image /= max(float(image.max()), 1e-12)

    rng = np.random.default_rng(seed)
    noisy = np.clip(
        image + rng.normal(0, 0.08, image.shape),
        0,
        1
    )
    return image, noisy


def load_image(upload):
    arr = np.asarray(
        Image.open(upload).convert("L"),
        dtype=np.float32
    )
    return arr / 255.0


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        raise ValueError("DICOM image contains no finite pixel values.")

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        raise ValueError("DICOM image has no usable intensity range.")

    return np.clip((image - low) / (high - low), 0, 1)


# -----------------------------
# DICOM loading
# -----------------------------
def load_dicom_series(uploaded_files):
    """
    Returns:
      raw_volume: original stored pixel values, used for pixel/count analysis
      display_volume: normalized display images, used for visualization
      metadata: selected DICOM metadata

    Note: stored pixel values are not automatically guaranteed to be
    radioactive counts. Check DICOM Units and the modality documentation.
    """
    if not DICOM_AVAILABLE:
        raise ImportError(
            "pydicom is not installed. Add pydicom to requirements.txt."
        )

    slices = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        ds = pydicom.dcmread(uploaded_file)

        if not hasattr(ds, "PixelData"):
            continue

        raw_pixels = np.asarray(ds.pixel_array)

        # This first prototype accepts single-frame 2D slices only.
        if raw_pixels.ndim != 2:
            continue

        raw_pixels = raw_pixels.astype(np.float32)

        # Use modality LUT/rescale only for display processing.
        display_pixels = apply_modality_lut(ds.pixel_array, ds)
        display_pixels = np.asarray(display_pixels, dtype=np.float32)

        if not np.all(np.isfinite(raw_pixels)):
            continue
        if not np.all(np.isfinite(display_pixels)):
            continue

        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            sort_position = float(position[2])
        else:
            sort_position = float(
                getattr(ds, "InstanceNumber", len(slices))
            )

        slices.append(
            (sort_position, raw_pixels, display_pixels, ds)
        )

    if not slices:
        raise ValueError(
            "No valid single-frame 2D DICOM image slices were found."
        )

    slices.sort(key=lambda item: item[0])

    raw_shapes = [item[1].shape for item in slices]
    display_shapes = [item[2].shape for item in slices]

    if len(set(raw_shapes)) != 1 or len(set(display_shapes)) != 1:
        raise ValueError("DICOM slices have different image dimensions.")

    raw_volume = np.stack([item[1] for item in slices])
    display_volume = np.stack([item[2] for item in slices])

    # Normalize display values only. Do not normalize raw_volume.
    display_volume = np.stack(
        [normalize_for_display(slice_image) for slice_image in display_volume]
    )

    first_ds = slices[0][3]
    metadata = {
        "Modality": str(getattr(first_ds, "Modality", "Unknown")),
        "Number of slices": len(slices),
        "Rows": int(getattr(first_ds, "Rows", 0)),
        "Columns": int(getattr(first_ds, "Columns", 0)),
        "Units": str(getattr(first_ds, "Units", "Not specified")),
        "Pixel representation": str(
            getattr(first_ds, "PixelRepresentation", "Not specified")
        ),
        "Rescale slope": str(getattr(first_ds, "RescaleSlope", "Not specified")),
        "Rescale intercept": str(
            getattr(first_ds, "RescaleIntercept", "Not specified")
        )
    }

    return raw_volume, display_volume, metadata


# -----------------------------
# Image quality calculations
# -----------------------------
def calculate_metrics(ref, candidate):
    mae = float(np.mean(np.abs(ref - candidate)))
    mse = float(np.mean((ref - candidate) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    return mae, mse, psnr


def calculate_snr(image):
    signal = float(np.mean(image))
    noise = float(np.std(image))

    if noise == 0:
        return float("inf")

    return float(
        20 * np.log10(max(abs(signal), 1e-12) / noise)
    )


def calculate_edge_strength(image):
    edge_x = sobel(image, axis=1)
    edge_y = sobel(image, axis=0)
    edge_magnitude = np.hypot(edge_x, edge_y)
    return float(np.mean(edge_magnitude))


def calculate_total_counts(count_image):
    return float(np.sum(count_image, dtype=np.float64))


def calculate_pixel_count(count_image, x, y):
    return float(count_image[y, x])


def calculate_roi_counts(count_image, x_start, y_start, roi_size):
    roi = count_image[
        y_start:y_start + roi_size,
        x_start:x_start + roi_size
    ]
    return float(np.sum(roi, dtype=np.float64))


# -----------------------------
# App interface
# -----------------------------
st.title("🛡️ NucleoGuard AI")
st.subheader("Nuclear Medicine Image Quality Assistant")
st.warning(
    "Research prototype only — not validated for clinical diagnosis."
)

modality = st.sidebar.selectbox(
    "Imaging Modality",
    ["PET", "SPECT"]
)
st.info(f"Current Imaging Modality: {modality}")

st.sidebar.header("Settings")

source = st.sidebar.radio(
    "Choose image source",
    [
        "Demo image",
        "Medical sample images",
        "Upload image",
        "Upload DICOM"
    ]
)

sigma = st.sidebar.slider(
    "Denoising strength",
    min_value=0.3,
    max_value=3.0,
    value=1.2,
    step=0.1
)

reference = None
raw_count_image = None
dicom_units = "Not available"
dicom_metadata = None


# -----------------------------
# Select and load input
# -----------------------------
if source == "Demo image":
    reference, noisy = create_demo_image()

elif source == "Medical sample images":
    try:
        reference = np.asarray(
            Image.open("FBP_Iter_single.jpg").convert("L"),
            dtype=np.float32
        ) / 255.0

        rng = np.random.default_rng(42)
        noisy = np.clip(
            reference + rng.normal(0, 0.08, reference.shape),
            0,
            1
        )
    except FileNotFoundError:
        st.error("Medical sample image FBP_Iter_single.jpg was not found.")
        st.stop()

elif source == "Upload image":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    noisy = load_image(uploaded)

    ref_file = st.file_uploader(
        "Optional: upload matching reference image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="reference_image"
    )

    if ref_file is not None:
        reference = load_image(ref_file)
        if reference.shape != noisy.shape:
            st.error(
                "Reference and test images must have the same dimensions."
            )
            st.stop()

elif source == "Upload DICOM":
    if not DICOM_AVAILABLE:
        st.error(
            "DICOM support is unavailable. Add pydicom to requirements.txt "
            "and redeploy the app."
        )
        st.stop()

    dicom_files = st.file_uploader(
        "Upload DICOM image slices from one series",
        type=None,
        accept_multiple_files=True
    )

    if not dicom_files:
        st.info("Upload the DICOM image files to begin. DICOMDIR alone is not enough.")
        st.stop()

    try:
        raw_volume, display_volume, dicom_metadata = load_dicom_series(
            dicom_files
        )

        st.success("DICOM image series loaded.")
        st.subheader("DICOM Study Information")

        for key, value in dicom_metadata.items():
            st.write(f"**{key}:** {value}")

        slice_index = st.slider(
            "Select DICOM slice",
            min_value=0,
            max_value=display_volume.shape[0] - 1,
            value=display_volume.shape[0] // 2
        )

        noisy = display_volume[slice_index]
        raw_count_image = raw_volume[slice_index]
        dicom_units = dicom_metadata["Units"]

        st.info(
            "The displayed image is normalized for visualization. "
            "Count analysis below uses the original stored DICOM pixel values."
        )

    except Exception as e:
        st.error(f"Could not read DICOM series: {e}")
        st.stop()


# -----------------------------
# Denoising and image indicators
# -----------------------------
enhanced = gaussian_filter(noisy, sigma=sigma)
change_map = np.abs(noisy - enhanced)

vmax = max(
    float(np.percentile(change_map, 99)),
    1e-6
)

st.write("Current denoising strength:", sigma)
st.write(
    "Mean absolute image change:",
    f"{float(np.mean(change_map)):.6f}"
)

snr_before = calculate_snr(noisy)
snr_after = calculate_snr(enhanced)

st.subheader("Experimental Image Quality Indicators")

col1, col2 = st.columns(2)
col1.metric("SNR Before", f"{snr_before:.2f} dB")
col2.metric(
    "SNR After",
    f"{snr_after:.2f} dB",
    delta=f"{snr_after - snr_before:+.2f} dB"
)

edge_before = calculate_edge_strength(noisy)
edge_after = calculate_edge_strength(enhanced)

edge_change = (
    (edge_after - edge_before) / max(edge_before, 1e-12)
) * 100

st.subheader("Edge Preservation Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Edge Strength Before", f"{edge_before:.4f}")
col2.metric("Edge Strength After", f"{edge_after:.4f}")
col3.metric("Edge Change", f"{edge_change:+.2f}%")


# -----------------------------
# Image display
# -----------------------------
st.header(f"{modality} Image Enhancement Results")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original Image")

axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Enhanced Image")

im = axes[2].imshow(
    change_map,
    cmap="hot",
    vmin=0,
    vmax=vmax
)
axes[2].set_title("Difference Map")

for ax in axes:
    ax.axis("off")

fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# -----------------------------
# ROI analysis
# -----------------------------
st.header("ROI Analysis")
st.write(
    "Select a square region of interest (ROI) to compare image "
    "intensity before and after denoising."
)

height, width = noisy.shape
roi_size = min(20, height, width)

col1, col2 = st.columns(2)

with col1:
    roi_x = st.slider(
        "ROI horizontal position",
        min_value=0,
        max_value=max(0, width - 1),
        value=width // 2,
        key="roi_x"
    )

with col2:
    roi_y = st.slider(
        "ROI vertical position",
        min_value=0,
        max_value=max(0, height - 1),
        value=height // 2,
        key="roi_y"
    )

x_start = min(
    max(0, roi_x - roi_size // 2),
    width - roi_size
)
y_start = min(
    max(0, roi_y - roi_size // 2),
    height - roi_size
)

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

# ROI preservation review
st.subheader("ROI Preservation Review")

roi_change = abs(after_mean - before_mean)

if abs(before_mean) > 1e-6:
    roi_change_percent = (roi_change / abs(before_mean)) * 100
else:
    roi_change_percent = 0.0

col1, col2 = st.columns(2)
col1.metric("Absolute ROI mean change", f"{roi_change:.4f}")
col2.metric("Relative ROI mean change", f"{roi_change_percent:.2f}%")

review_limit = st.slider(
    "ROI change review threshold (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

if roi_change_percent >= review_limit:
    st.warning(
        "Review recommended: the ROI mean intensity changed beyond "
        "the selected threshold."
    )
else:
    st.info(
        "The ROI mean change is below the selected threshold. "
        "This does not guarantee preservation of diagnostic details."
    )

st.caption(
    "Experimental screening indicator only. "
    "The threshold is user-defined and is not clinically validated."
)


# -----------------------------
# Counts analysis
# -----------------------------
st.header("Counts Analysis")

if raw_count_image is None:
    st.warning(
        "True radioactive counts cannot be calculated from the current "
        "Demo, JPG, PNG, or other normalized image input. Upload original "
        "DICOM image data to inspect stored pixel values."
    )
    st.info(
        "For non-DICOM images, pixel intensity values are not radioactive "
        "counts. They must not be reported as Total Counts or ROI Counts."
    )
else:
    units_upper = str(dicom_units).strip().upper()

    # Pixel coordinates for the selected pixel.
    count_col1, count_col2 = st.columns(2)

    with count_col1:
        pixel_x = st.slider(
            "Pixel X coordinate",
            min_value=0,
            max_value=raw_count_image.shape[1] - 1,
            value=raw_count_image.shape[1] // 2,
            key="count_pixel_x"
        )

    with count_col2:
        pixel_y = st.slider(
            "Pixel Y coordinate",
            min_value=0,
            max_value=raw_count_image.shape[0] - 1,
            value=raw_count_image.shape[0] // 2,
            key="count_pixel_y"
        )

    total_stored_values = calculate_total_counts(raw_count_image)
    selected_pixel_value = calculate_pixel_count(
        raw_count_image, pixel_x, pixel_y
    )
    roi_stored_sum = calculate_roi_counts(
        raw_count_image, x_start, y_start, roi_size
    )

    count_col1, count_col2, count_col3 = st.columns(3)

    if units_upper in ("CNTS", "COUNTS"):
        count_col1.metric("Total Counts", f"{total_stored_values:,.0f}")
        count_col3.metric("ROI Counts", f"{roi_stored_sum:,.0f}")
        st.success(
            "The DICOM Units field indicates counts (CNTS/COUNTS). These values are sums of "
            "stored pixel values for the selected slice and ROI."
        )
    else:
        count_col1.metric(
            "Total Stored Pixel Values",
            f"{total_stored_values:,.2f}"
        )
        count_col3.metric(
            "ROI Stored Pixel-value Sum",
            f"{roi_stored_sum:,.2f}"
        )
        st.warning(
            f"DICOM Units are '{dicom_units}', not explicitly COUNTS. "
            "These are stored pixel-value sums and must not be interpreted "
            "as calibrated radioactive counts."
        )

    count_col2.metric(
        "Selected Pixel Value",
        f"{selected_pixel_value:,.2f}"
    )

    st.caption(
        f"Selected pixel coordinates: X={pixel_x}, Y={pixel_y}. "
        f"ROI coordinates: X={x_start} to {x_start + roi_size - 1}, "
        f"Y={y_start} to {y_start + roi_size - 1}."
    )

    st.caption(
        "Counts are calculated from the original stored pixel values, "
        "not from the normalized display image or denoised image. "
        "Interpretation depends on the DICOM modality, Units, and "
        "acquisition/reconstruction settings."
    )


# -----------------------------
# Change detection
# -----------------------------
st.header("Image Comparison and Change Detection")

threshold = float(np.percentile(change_map, 95))
high_change = change_map >= threshold

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

fig.colorbar(
    heatmap,
    ax=axes[2],
    fraction=0.046,
    pad=0.04,
    label="Absolute intensity change"
)

for ax in axes:
    ax.axis("off")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Cyan outlines mark pixels in the highest 5% of observed changes. "
    "This is a relative change indicator, not a calibrated uncertainty map."
)

st.subheader("Change Detection Alert")

change_limit = st.slider(
    "Minimum intensity change",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01
)

changed_pixels = change_map > change_limit
change_percentage = (
    np.count_nonzero(changed_pixels) / changed_pixels.size
) * 100

col1, col2 = st.columns(2)
col1.metric("Pixels with significant change", f"{change_percentage:.2f}%")
col2.metric("Change threshold", f"{change_limit:.2f}")

if change_percentage >= 10:
    st.warning(
        "Review recommended: a considerable portion of the image "
        "changed during enhancement."
    )
else:
    st.success(
        "No large-scale change detected at the selected threshold. "
        "This does not guarantee preservation of all diagnostic details."
    )

st.caption(
    "This is an experimental image-change indicator, not a validated "
    "measure of diagnostic accuracy or uncertainty."
)


# -----------------------------
# Reference-based metrics
# -----------------------------
if reference is not None and reference.shape == enhanced.shape:
    st.header("Reference-based Image Metrics")

    mae, mse, psnr = calculate_metrics(reference, enhanced)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.6f}")
    col3.metric("PSNR", f"{psnr:.2f} dB")

    st.caption(
        "These metrics compare the processed image with the available "
        "reference image. They are experimental and do not establish "
        "clinical validity."
    )


# -----------------------------
# Downloadable report
# -----------------------------
report = f"""
NUCLEOGUARD AI
Image Quality Analysis Report

Imaging Modality: {modality}
Image Source: {source}
Denoising Strength: {sigma}

SNR Before: {snr_before:.4f} dB
SNR After: {snr_after:.4f} dB

Edge Strength Before: {edge_before:.4f}
Edge Strength After: {edge_after:.4f}
Edge Change: {edge_change:+.2f}%

ROI Size: {roi_size} x {roi_size}
ROI Mean Before: {before_mean:.6f}
ROI Mean After: {after_mean:.6f}
ROI Mean Change: {roi_change:.6f}
ROI Relative Change: {roi_change_percent:.2f}%

DICOM Units: {dicom_units}
"""

if raw_count_image is not None:
    report += f"""
Total Stored Pixel Values: {calculate_total_counts(raw_count_image):.4f}
Selected Pixel Value: {calculate_pixel_count(raw_count_image, pixel_x, pixel_y):.4f}
ROI Stored Pixel-value Sum: {calculate_roi_counts(raw_count_image, x_start, y_start, roi_size):.4f}
"""

report += """
Important:
This report is generated by an experimental research prototype.
It is not validated for clinical diagnosis or treatment decisions.
Pixel-value sums should only be interpreted as radioactive counts
when the DICOM Units and acquisition context support that interpretation.
"""

st.subheader("Analysis Summary Report")
st.text(report)

st.download_button(
    label="Download Analysis Report",
    data=report,
    file_name="NucleoGuard_Analysis_Report.txt",
    mime="text/plain"
)
