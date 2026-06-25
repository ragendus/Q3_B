import os
import io
import time
import glob
import subprocess
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fingerprint import FingerprintIndex

INDEX_PATH = "fingerprint_index.pkl"
SAMPLES_DIR = "samples"   # optional folder of demo clips shown as "Try" buttons
SONGS_DIR = "songs"       # where the 50-song library gets downloaded to
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1UV96lMDwvP-N5Zur6tOPx5OwA6vaDs8N"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

st.set_page_config(page_title="EE200: Audio Fingerprinting", page_icon="🎵", layout="wide")


# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_index(path):
    """
    Load the prebuilt index if it's already in the repo. Otherwise, on a
    fresh Streamlit Cloud deployment, download the 50-song library from the
    public Drive folder and build the index once (cached for the life of
    this running instance).
    """
    if os.path.exists(path):
        return FingerprintIndex.load(path)

    with st.spinner("First run: downloading library and building fingerprint index... "
                     "this can take a few minutes, but only happens once."):
        os.makedirs(SONGS_DIR, exist_ok=True)
        subprocess.run(
            ["gdown", "--folder", DRIVE_FOLDER_URL, "-O", SONGS_DIR],
            check=False,
        )

        paths = []
        for ext in AUDIO_EXTS:
            paths.extend(glob.glob(os.path.join(SONGS_DIR, "**", f"*{ext}"), recursive=True))
        paths = sorted(set(paths))

        if not paths:
            return None

        idx = FingerprintIndex()
        progress = st.progress(0.0)
        for i, p in enumerate(paths):
            try:
                idx.index_song(p)
            except Exception as e:
                st.warning(f"Skipped {os.path.basename(p)}: {e}")
            progress.progress((i + 1) / len(paths))
        idx.save(path)
        return idx


def constellation_thumb(peaks, color="#36e0c4"):
    """Small dark-background scatter plot used for Library thumbnails."""
    fig, ax = plt.subplots(figsize=(2.6, 1.6), dpi=120)
    fig.patch.set_facecolor("#0b0f10")
    ax.set_facecolor("#0b0f10")
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=2, c=color, alpha=0.85, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    return fig


def spectrogram_fig(Sxx_db):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.imshow(Sxx_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("freq bin")
    fig.tight_layout()
    return fig


def constellation_fig(peaks):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    fig.patch.set_facecolor("#0b0f10")
    ax.set_facecolor("#0b0f10")
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=3, c="#5ad1ff", linewidths=0)
    ax.set_xlabel("time (frames)", color="white")
    ax.set_ylabel("freq bin", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.tight_layout()
    return fig


def offset_histogram_fig(hist_dict):
    """
    Bar chart of vote-count per time offset for the winning song - this is
    the 'alignment spike' plot: one thin tall spike (the true offset, where
    every matched hash agrees) standing far above a flat noise floor of
    chance matches scattered at random offsets, with a diagonal label
    pointing at the spike (mirrors the reference demo video).
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    fig.patch.set_facecolor("#0b0f10")
    ax.set_facecolor("#0b0f10")

    if hist_dict:
        offsets = np.array(sorted(hist_dict.keys()))
        counts = np.array([hist_dict[o] for o in offsets])
        best_idx = np.argmax(counts)
        x_span = max(offsets.max() - offsets.min(), 1)

        # flat noise floor: every non-winning offset, drawn as a thin near-zero line
        ax.vlines(offsets, 0, counts, color="#3a7ca5", linewidth=1, alpha=0.6)
        # the spike itself: one thin bright vertical line, much taller than the rest
        ax.vlines(offsets[best_idx], 0, counts[best_idx], color="#ffa53e", linewidth=2)

        ax.annotate(
            f"{counts[best_idx]:,} hashes\nalign here",
            xy=(offsets[best_idx], counts[best_idx]),
            xytext=(offsets[best_idx] + x_span * 0.12, counts[best_idx] * 0.62),
            color="#ffa53e", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#ffa53e", linewidth=1),
        )
        ax.set_ylim(0, counts[best_idx] * 1.15)

    ax.set_xlabel("offset (frames)", color="white")
    ax.set_ylabel("# hashes", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    fig.tight_layout()
    return fig


def save_upload_to_tmp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ----------------------------------------------------------------------------
st.markdown("## 🎵 EE200: Audio Fingerprinting")
st.caption("SIGNALS, SYSTEMS & NETWORKS · PROJECT DEMO")
st.write("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

index = load_index(INDEX_PATH)

if index is None:
    st.error(
        "Could not build a fingerprint index - no audio files were found after "
        f"downloading `{DRIVE_FOLDER_URL}`. Check that the Drive folder is still "
        "public, or commit a prebuilt `fingerprint_index.pkl` to the repo instead."
    )
    st.stop()

tab_library, tab_identify, tab_batch = st.tabs(["◈ LIBRARY", "⊙ IDENTIFY", "▤ BATCH"])

# ============================================================ LIBRARY ======
with tab_library:
    st.info("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
    st.markdown("**IN THE DATABASE**")
    cols = st.columns(4)
    for i, (song_id, meta) in enumerate(index.songs.items()):
        with cols[i % 4]:
            st.pyplot(constellation_thumb(meta["peak_sample"]), use_container_width=True)
            st.markdown(f"**{meta['name']}**")
            st.caption(f"{meta['n_hashes']:,} hashes")

# ============================================================ IDENTIFY =====
with tab_identify:
    st.markdown("**SEARCH**")
    st.markdown("### Identify a clip")

    uploaded = st.file_uploader(
        "Upload", type=["wav", "mp3", "flac", "ogg", "m4a"],
        label_visibility="visible", help="200MB per file"
    )

    query_path = None
    if uploaded is not None:
        query_path = save_upload_to_tmp(uploaded)

    st.markdown("**OR TRY A SAMPLE**")
    if os.path.isdir(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            fpath = os.path.join(SAMPLES_DIR, fname)
            c1, c2 = st.columns([5, 1])
            with c1:
                st.audio(fpath)
            with c2:
                if st.button("Try", key=f"try_{fname}"):
                    query_path = fpath

    if st.button("Identify", type="primary") and query_path:
        with st.spinner("Running pipeline..."):
            result = index.identify(query_path)
        timing = result["timing"]

        # --- timing strip -------------------------------------------------
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("① SPECTROGRAM", f"{timing['spectrogram_ms']:.0f} ms",
                   f"{timing['spectrogram_shape'][0]}×{timing['spectrogram_shape'][1]}")
        t2.metric("② CONSTELLATION", f"{timing['constellation_ms']:.0f} ms",
                   f"{timing['n_peaks']} peaks")
        t3.metric("③ HASHING", f"{timing['hashing_ms']:.0f} ms",
                   f"{timing['n_hashes']:,} hashes")
        t4.metric("④ DB LOOKUP", f"{timing['lookup_ms']:.0f} ms",
                   f"{timing['n_tracks_touched']} tracks")
        t5.metric("⑤ SCORING", f"{timing['scoring_ms']:.0f} ms",
                   f"offset {result['candidates'][0]['offset'] if result['candidates'] else '-'}")
        t6.metric("TOTAL", f"{timing['total_ms']:.0f} ms")

        # --- match card -----------------------------------------------------
        if result["match"]:
            m = result["match"]
            ratio_txt = f"{m['runner_up_ratio']:.0f}x the runner-up" if m["runner_up_ratio"] != float("inf") else "no runner-up"
            st.success(f"**MATCH FOUND**\n\n# {m['name']}\n\ncluster score **{m['score']}** · {ratio_txt}")
        else:
            st.warning("No match found above the confidence threshold.")

        # --- candidate bar list ---------------------------------------------
        st.markdown("**CANDIDATE SCORES**")
        if result["candidates"]:
            max_score = result["candidates"][0]["score"]
            for c in result["candidates"]:
                bcol, scol = st.columns([6, 1])
                with bcol:
                    st.progress(min(c["score"] / max_score, 1.0) if max_score else 0, text=c["name"])
                with scol:
                    st.write(c["score"])

        # --- offset histogram: the alignment spike ---------------------------
        st.markdown("**STEP 3 · THE PROOF**")
        st.markdown("##### The alignment spike")
        if result["winning_histogram"] and result["match"]:
            best_count = result["match"]["score"]
            st.write(
                f"Every matched hash votes for a time offset (database frame minus query frame). "
                f"Chance matches scatter votes randomly, forming a flat noise floor. A genuine match "
                f"makes them converge: **{best_count:,} hashes agreed on a single offset**. "
                f"That spike cannot be a coincidence."
            )
            st.pyplot(offset_histogram_fig(result["winning_histogram"]), use_container_width=True)
        else:
            st.caption("No matching hashes were found, so no offset histogram could be built.")

        # --- spectrogram + constellation -------------------------------------
        st.markdown("**STEP 1 · FEATURE EXTRACTION**")
        st.markdown("##### From spectrogram to constellation")
        st.write(
            f"The clip was converted into a time-frequency map (left); brighter means louder. "
            f"From that, only the **{timing['n_peaks']} most prominent peaks** were kept (right)."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(spectrogram_fig(result["spectrogram"]), use_container_width=True)
        with c2:
            st.pyplot(constellation_fig(result["peaks"]), use_container_width=True)

# ============================================================ BATCH ========
with tab_batch:
    st.markdown("**BATCH**")
    st.markdown("### Identify many clips at once")
    st.write(
        "Upload a set of query clips. Each is identified against the **currently indexed library**, "
        "and the results are written to a standardised `results.csv` with columns `filename, prediction`. "
        "The `prediction` is the matched track's filename without its extension, or `none` when no "
        "candidate clears the confidence threshold."
    )

    batch_files = st.file_uploader(
        "Upload", type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True, key="batch_uploader"
    )

    if st.button("Run batch", type="primary") and batch_files:
        rows = []
        progress = st.progress(0.0)
        for i, f in enumerate(batch_files):
            path = save_upload_to_tmp(f)
            result = index.identify(path)
            if result["match"] and index.is_confident(result):
                prediction = result["match"]["name"]
            else:
                prediction = "none"
            rows.append({"filename": f.name, "prediction": prediction})
            progress.progress((i + 1) / len(batch_files))

        df = pd.DataFrame(rows)
        st.markdown("**RESULTS**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        n_matched = (df["prediction"] != "none").sum()
        st.caption(f"{n_matched} / {len(df)} clips matched to a track "
                   f"({len(df) - n_matched} returned `none`).")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", data=csv_bytes,
                            file_name="results.csv", mime="text/csv")
