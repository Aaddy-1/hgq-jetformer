#!/usr/bin/env python3
"""Resumable, parallel fetch of the two JetClass archives needed to rebuild test.h5.

`download_jetclass.py` uses `urllib.request.urlretrieve` (:38): a single stream
with no resume, no retry and no timeout, so a dropped connection restarts the
whole 28 GB transfer. It also verifies MD5 only on the already-on-disk path
(:19-28) -- a file downloaded fresh is never hashed. And because
`extract_archive` deletes each tarball after extraction (:150), re-running it
re-pulls all twelve archives (~250 GB) when only two are damaged.

This fetches only `test_20M` and `val_5M` (~35 GB). Each file is split into
byte ranges downloaded concurrently; completed ranges are recorded in a
`.ranges` sidecar so an interrupted run resumes where it stopped rather than at
zero. MD5 is checked before the file is declared good.

The URL and hash tables are imported from `download_jetclass.py` rather than
copied, so there is one source of truth for what "correct" means.

Whether parallel ranges actually help depends on how Zenodo throttles -- per
connection (they help a great deal) or per client IP (they do not). The script
is correct either way; only the wall clock changes.

Usage:
    python -m scripts.fetch_jetclass_test
    python -m scripts.fetch_jetclass_test --connections 8 --chunk_mb 32
    python -m scripts.fetch_jetclass_test --verify_only
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download_jetclass import datasets as JETCLASS_DATASETS  # noqa: E402

# The "Pythia/" group holds exactly the test_20M and val_5M archives; the
# train_100M group is deliberately not fetched.
TEST_ARCHIVES = list(JETCLASS_DATASETS["JetClass"]["Pythia/"])


def md5_of_file(fpath, block=8 << 20):
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(url, timeout=60):
    """Returns (size_in_bytes, server_supports_ranges)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        size = int(r.headers.get("Content-Length") or 0)
        accepts = (r.headers.get("Accept-Ranges") or "").lower() == "bytes"
    return size, accepts


def _fetch_range(url, fpath, index, start, end, timeout, attempts):
    """Streams one byte range into the preallocated file at its true offset.

    A failed attempt may leave a partially written range, which is harmless:
    the range is not recorded as done, so it is fetched again in full.
    """
    want = end - start + 1
    last_err = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            got = 0
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 206:
                    raise IOError(f"expected 206 partial content, got {r.status}")
                with open(fpath, "r+b") as f:
                    f.seek(start)
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
                        got += len(block)
            if got != want:
                raise IOError(f"short range: {got} of {want} bytes")
            return index, want, None
        except Exception as e:  # noqa: BLE001 - any failure is retried identically
            last_err = f"{type(e).__name__}: {e}"
            if attempt < attempts - 1:
                time.sleep(min(30, 2**attempt))
    return index, 0, last_err


def _sidecar_path(fpath):
    return fpath + ".ranges"


def _load_sidecar(fpath, size, chunk_bytes):
    """Returns the set of completed range indices, or an empty set if unusable."""
    p = _sidecar_path(fpath)
    if not os.path.exists(p) or not os.path.exists(fpath):
        return set()
    try:
        with open(p) as f:
            state = json.load(f)
    except Exception:
        return set()
    # A different size or chunking makes recorded indices meaningless.
    if state.get("size") != size or state.get("chunk") != chunk_bytes:
        return set()
    if os.path.getsize(fpath) != size:
        return set()
    return set(state.get("done", []))


def _save_sidecar(fpath, size, chunk_bytes, done):
    tmp = _sidecar_path(fpath) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"size": size, "chunk": chunk_bytes, "done": sorted(done)}, f)
    os.replace(tmp, _sidecar_path(fpath))


def _download_single_stream(url, fpath, size, timeout):
    """Fallback for a server that will not serve ranges: one stream, resumed by offset."""
    have = os.path.getsize(fpath) if os.path.exists(fpath) else 0
    if have >= size > 0:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "r+b" if have else "wb"
    with urllib.request.urlopen(req, timeout=timeout) as r, open(fpath, mode) as f:
        if have:
            f.seek(have)
        done = have
        t0 = time.time()
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            elapsed = max(1e-6, time.time() - t0)
            rate = (done - have) / elapsed / (1 << 20)
            pct = 100.0 * done / size if size else 0.0
            print(
                f"\r  {done / (1 << 30):.2f} / {size / (1 << 30):.2f} GiB "
                f"({pct:5.1f}%)  {rate:6.1f} MiB/s",
                end="",
                flush=True,
            )
    print()


def fetch_archive(url, expected_md5, out_dir, connections, chunk_mb, timeout, attempts):
    fname = os.path.basename(url)
    fpath = os.path.join(out_dir, fname)

    if os.path.exists(fpath):
        print(f"[{fname}] present, hashing before re-downloading...")
        if md5_of_file(fpath) == expected_md5:
            print(f"[{fname}] MD5 OK ({expected_md5}) -- nothing to do")
            return True
        print(f"[{fname}] MD5 mismatch; resuming/redownloading")

    size, accepts_ranges = probe(url, timeout=timeout)
    if size <= 0:
        print(f"[{fname}] server did not report a size; cannot range-split")
        accepts_ranges = False

    if not accepts_ranges:
        print(f"[{fname}] server will not serve ranges -- falling back to one stream")
        _download_single_stream(url, fpath, size, timeout)
        ok = md5_of_file(fpath) == expected_md5
        print(f"[{fname}] MD5 {'OK' if ok else 'MISMATCH'}")
        return ok

    chunk_bytes = chunk_mb << 20
    n_chunks = (size + chunk_bytes - 1) // chunk_bytes
    done = _load_sidecar(fpath, size, chunk_bytes)

    if not done:
        # Preallocate so every worker can seek to its own offset independently.
        with open(fpath, "wb") as f:
            f.truncate(size)
        done = set()

    todo = [i for i in range(n_chunks) if i not in done]
    print(
        f"[{fname}] {size / (1 << 30):.2f} GiB in {n_chunks} x {chunk_mb} MiB ranges; "
        f"{len(done)} already done, {len(todo)} to fetch, {connections} connections"
    )
    if not todo:
        print(f"[{fname}] all ranges present; verifying")
    else:
        t0 = time.time()
        bytes_done = 0
        failures = []
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = {}
            for i in todo:
                start = i * chunk_bytes
                end = min(start + chunk_bytes, size) - 1
                futures[
                    pool.submit(
                        _fetch_range, url, fpath, i, start, end, timeout, attempts
                    )
                ] = i
            for n, fut in enumerate(as_completed(futures), start=1):
                index, nbytes, err = fut.result()
                if err:
                    failures.append((index, err))
                else:
                    done.add(index)
                    bytes_done += nbytes
                if n % 8 == 0 or n == len(todo):
                    _save_sidecar(fpath, size, chunk_bytes, done)
                    elapsed = max(1e-6, time.time() - t0)
                    rate = bytes_done / elapsed / (1 << 20)
                    remaining = (len(todo) - n) * chunk_bytes / (1 << 20)
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"\r  {n}/{len(todo)} ranges  "
                        f"{bytes_done / (1 << 30):6.2f} GiB  "
                        f"{rate:6.1f} MiB/s  ETA {eta / 60:5.1f} min",
                        end="",
                        flush=True,
                    )
        print()
        _save_sidecar(fpath, size, chunk_bytes, done)
        if failures:
            print(f"[{fname}] {len(failures)} ranges failed after {attempts} attempts:")
            for index, err in failures[:5]:
                print(f"    range {index}: {err[:120]}")
            print(f"[{fname}] re-run to retry only the missing ranges")
            return False

    print(f"[{fname}] verifying MD5 ({size / (1 << 30):.2f} GiB)...")
    actual = md5_of_file(fpath)
    if actual != expected_md5:
        print(f"[{fname}] MD5 MISMATCH: got {actual}, expected {expected_md5}")
        print(f"[{fname}] delete {fpath} and {_sidecar_path(fpath)} and re-run")
        return False

    print(f"[{fname}] MD5 OK ({expected_md5})")
    if os.path.exists(_sidecar_path(fpath)):
        os.remove(_sidecar_path(fpath))
    return True


def main(out_dir, connections, chunk_mb, timeout, attempts, verify_only):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for url, expected_md5 in TEST_ARCHIVES:
        fname = os.path.basename(url)
        fpath = os.path.join(out_dir, fname)

        if verify_only:
            if not os.path.exists(fpath):
                print(f"[{fname}] absent")
                results.append(False)
                continue
            ok = md5_of_file(fpath) == expected_md5
            print(f"[{fname}] MD5 {'OK' if ok else 'MISMATCH'}")
            results.append(ok)
            continue

        results.append(
            fetch_archive(
                url, expected_md5, out_dir, connections, chunk_mb, timeout, attempts
            )
        )

    if all(results):
        print(f"\nAll {len(results)} archives verified in {out_dir}")
        print("Next: extract into datasets/JetClass/Pythia/, then")
        print("      python -m scripts.rebuild_test_h5 --dry_run")
        return 0

    print(f"\n{results.count(False)} of {len(results)} archives incomplete -- re-run")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch only the JetClass test_20M and val_5M archives"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="datasets/JetClass",
        help="Where the .tar files are written (default: datasets/JetClass)",
    )
    parser.add_argument(
        "-c",
        "--connections",
        type=int,
        default=16,
        help="Concurrent range requests per archive (default: 16)",
    )
    parser.add_argument(
        "--chunk_mb",
        type=int,
        default=64,
        help="Size of each byte range in MiB (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request socket timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=6,
        help="Retries per range before giving up (default: 6)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Hash what is already on disk and exit without downloading",
    )
    args = parser.parse_args()

    sys.exit(
        main(
            out_dir=args.out_dir,
            connections=args.connections,
            chunk_mb=args.chunk_mb,
            timeout=args.timeout,
            attempts=args.attempts,
            verify_only=args.verify_only,
        )
    )
