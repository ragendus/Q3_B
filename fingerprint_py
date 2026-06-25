import os
import time
import pickle
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter

try:
    import librosa
    _HAVE_LIBROSA = True
except ImportError:
    _HAVE_LIBROSA = False


# ----------------------------------------------------------------------------
# Tunable parameters (kept as module-level constants so app.py can show them)
# ----------------------------------------------------------------------------
SR = 22050              # working sample rate (everything is resampled to this)
N_FFT = 2048             # window size for the STFT
HOP = N_FFT // 4         # hop length between STFT frames
PEAK_NEIGHBORHOOD = (20, 20)   # (freq_cells, time_cells) for local-max filter
AMP_MIN_DB = -55         # ignore peaks quieter than this (noise floor)
MAX_PEAKS_PER_SONG = 6000      # cap peaks for very long songs (keeps loudest)

FAN_OUT = 8              # how many partner-peaks each anchor peak pairs with
MIN_DT = 1               # minimum frame gap between anchor and partner
MAX_DT = 200             # maximum frame gap between anchor and partner (target zone width)

# bit widths used when packing a hash into a single integer
F_BITS = 10
DT_BITS = 8
F_MASK = (1 << F_BITS) - 1
DT_MASK = (1 << DT_BITS) - 1


def _load_audio(path, sr=SR):
    """Load any common audio file as a mono float32 waveform at `sr` Hz."""
    if _HAVE_LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    # Fallback: soundfile + manual resample (rarely needed, librosa covers WAV/MP3/FLAC/OGG/M4A)
    import soundfile as sf
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import scipy.signal as sig
        n_target = int(len(y) * sr / file_sr)
        y = sig.resample(y, n_target)
    return y.astype(np.float32)


def compute_spectrogram(y, sr=SR):
    """Return (freqs, times, Sxx_db) - magnitude spectrogram in dB."""
    freqs, times, Sxx = scipy_spectrogram(
        y, fs=sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, mode="magnitude"
    )
    Sxx_db = 20 * np.log10(Sxx + 1e-6)
    return freqs, times, Sxx_db


def find_peaks(Sxx_db, amp_min_db=AMP_MIN_DB, neighborhood=PEAK_NEIGHBORHOOD, max_peaks=None):
    """
    Constellation map: keep time-frequency bins that are a local maximum in
    their neighborhood AND louder than amp_min_db.
    Returns an (N, 2) int array of (freq_bin, time_bin) peak coordinates.
    """
    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    loud_enough = Sxx_db > amp_min_db
    peak_mask = local_max & loud_enough
    f_idx, t_idx = np.nonzero(peak_mask)

    if max_peaks is not None and len(f_idx) > max_peaks:
        # keep the loudest `max_peaks` peaks
        amps = Sxx_db[f_idx, t_idx]
        keep = np.argsort(amps)[-max_peaks:]
        f_idx, t_idx = f_idx[keep], t_idx[keep]

    # sort by time so the fan-out pairing below can scan forward efficiently
    order = np.argsort(t_idx)
    return np.stack([f_idx[order], t_idx[order]], axis=1)


def generate_hashes(peaks, fan_out=FAN_OUT, min_dt=MIN_DT, max_dt=MAX_DT):
    """
    Pair each anchor peak with up to `fan_out` nearby peaks that come later
    in time (within [min_dt, max_dt] frames). Each pair becomes one hash.

    Returns a list of (hash_int, t_anchor) tuples.
    """
    hashes = []
    n = len(peaks)
    times = peaks[:, 1]

    for i in range(n):
        f1, t1 = peaks[i]
        # peaks are time-sorted, so partners live in a forward window
        j = i + 1
        paired = 0
        while j < n and paired < fan_out:
            f2, t2 = peaks[j]
            dt = t2 - t1
            if dt > max_dt:
                break
            if dt >= min_dt:
                h = ((int(f1) & F_MASK) << (F_BITS + DT_BITS)) \
                    | ((int(f2) & F_MASK) << DT_BITS) \
                    | (int(dt) & DT_MASK)
                hashes.append((h, int(t1)))
                paired += 1
            j += 1
    return hashes


class FingerprintIndex:
    """
    In-memory database:
        self.db[hash]        -> list of (song_id, t_anchor)
        self.songs[song_id]  -> {"name", "path", "n_hashes", "peaks"(small sample for thumbnail)}
    Persist with .save(path) / FingerprintIndex.load(path).
    """

    def __init__(self):
        self.db = {}
        self.songs = {}
        self._next_id = 0

    # -- building -------------------------------------------------------
    def index_song(self, path, name=None):
        name = name or os.path.splitext(os.path.basename(path))[0]
        y = _load_audio(path)
        _, _, Sxx_db = compute_spectrogram(y)
        peaks = find_peaks(Sxx_db, max_peaks=MAX_PEAKS_PER_SONG)
        hashes = generate_hashes(peaks)

        song_id = self._next_id
        self._next_id += 1

        for h, t in hashes:
            self.db.setdefault(h, []).append((song_id, t))

        # keep a small random sample of peaks just for drawing a thumbnail later
        sample = peaks
        if len(sample) > 800:
            idx = np.random.choice(len(sample), 800, replace=False)
            sample = sample[idx]

        self.songs[song_id] = {
            "name": name,
            "path": path,
            "n_hashes": len(hashes),
            "n_peaks": len(peaks),
            "peak_sample": sample,
        }
        return song_id

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"db": self.db, "songs": self.songs, "next_id": self._next_id}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.db = data["db"]
        idx.songs = data["songs"]
        idx._next_id = data["next_id"]
        return idx

    # -- querying ---------------------------------------------------------
    def identify(self, query_path, top_k=5):
        """
        Run the full pipeline on a query clip and return a result dict with
        per-step timing (ms) plus ranked candidate scores, mirroring the
        "MATCH FOUND" / timing-strip UI in the reference screenshots.
        """
        timing = {}
        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        y = _load_audio(query_path)
        freqs, times, Sxx_db = compute_spectrogram(y)
        timing["spectrogram_ms"] = (time.perf_counter() - t0) * 1000
        timing["spectrogram_shape"] = Sxx_db.shape

        t0 = time.perf_counter()
        peaks = find_peaks(Sxx_db)
        timing["constellation_ms"] = (time.perf_counter() - t0) * 1000
        timing["n_peaks"] = len(peaks)

        t0 = time.perf_counter()
        hashes = generate_hashes(peaks)
        timing["hashing_ms"] = (time.perf_counter() - t0) * 1000
        timing["n_hashes"] = len(hashes)

        t0 = time.perf_counter()
        # offset[song_id][delta] = vote count   (delta = db_time - query_time)
        offset_votes = {}
        for h, t_query in hashes:
            matches = self.db.get(h)
            if not matches:
                continue
            for song_id, t_db in matches:
                delta = t_db - t_query
                d = offset_votes.setdefault(song_id, {})
                d[delta] = d.get(delta, 0) + 1
        timing["lookup_ms"] = (time.perf_counter() - t0) * 1000
        timing["n_tracks_touched"] = len(offset_votes)

        t0 = time.perf_counter()
        scored = []
        for song_id, hist in offset_votes.items():
            best_delta = max(hist, key=hist.get)
            best_score = hist[best_delta]
            scored.append({
                "song_id": song_id,
                "name": self.songs[song_id]["name"],
                "score": best_score,
                "offset": best_delta,
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        timing["scoring_ms"] = (time.perf_counter() - t0) * 1000

        timing["total_ms"] = (time.perf_counter() - t_total0) * 1000

        top = scored[:top_k]
        match = None
        winning_histogram = None
        if top:
            top_score = top[0]["score"]
            runner_up = top[1]["score"] if len(top) > 1 else 0
            match = {
                "name": top[0]["name"],
                "score": top_score,
                "runner_up_ratio": (top_score / runner_up) if runner_up > 0 else float("inf"),
            }
            # full offset -> vote-count histogram for the winning song, used to
            # draw the "alignment spike" plot (one tall bar among a flat noise floor)
            winning_histogram = offset_votes[top[0]["song_id"]]

        return {
            "timing": timing,
            "candidates": top,
            "match": match,
            "spectrogram": Sxx_db,
            "peaks": peaks,
            "winning_histogram": winning_histogram,
        }

    def is_confident(self, result, min_score=15, min_ratio=1.5):
        """Simple confidence gate used by the Batch tab to decide `none`."""
        m = result["match"]
        if m is None:
            return False
        return m["score"] >= min_score and m["runner_up_ratio"] >= min_ratio
