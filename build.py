import sys
import os
import glob
from fingerprint import FingerprintIndex

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def main():
    songs_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/songs"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fingerprint_index.pkl"

    paths = []
    for ext in AUDIO_EXTS:
        paths.extend(glob.glob(os.path.join(songs_dir, "**", f"*{ext}"), recursive=True))
    paths = sorted(set(paths))

    if not paths:
        print(f"No audio files found in {songs_dir}")
        sys.exit(1)

    print(f"Found {len(paths)} audio files. Indexing...")
    idx = FingerprintIndex()
    for i, p in enumerate(paths, 1):
        try:
            idx.index_song(p)
            print(f"  [{i}/{len(paths)}] indexed: {os.path.basename(p)}  "
                  f"({idx.songs[i-1]['n_hashes']} hashes)")
        except Exception as e:
            print(f"  [{i}/{len(paths)}] FAILED on {p}: {e}")

    idx.save(out_path)
    print(f"\nSaved index with {len(idx.songs)} songs -> {out_path}")


if __name__ == "__main__":
    main()
