"""Download a dataset and create its fixed splits.

    uv run python -m near_duplicates.download caltech101           # 130 MB, about a minute
    uv run python -m near_duplicates.download disc21               # about 12 GB of selected images
    uv run python -m near_duplicates.download disc21 --limit 200   # smoke test

DISC21 images are fetched one by one from Meta's zip archives with HTTP range requests, so only the images
named in the splits are downloaded, never whole 8.8 GB archives. Reruns skip finished files, so an
interrupted download resumes. DISC21's licence forbids redistributing the images: never commit `data/`.
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import tarfile
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from near_duplicates.datasets import (
    CALTECH_ROOT,
    DISC21_DIR,
    caltech_image_ids,
    dev_ground_truth,
    image_path,
    load_split,
    make_caltech_splits,
    make_disc21_splits,
    splits_exist,
    write_splits,
)

CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
DISC21_URL = "https://dl.fbaipublicfiles.com/image_similarity_challenge/public"
DISC21_IMAGES_PER_REFERENCE_ZIP = 50_000
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
HEADER_SLACK = 512
RETRIES = 3


def download_file(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as out, tqdm(total=total, unit="B", unit_scale=True, desc=target.name, file=sys.stderr) as bar:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, target)
    return target


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, so `zipfile` can read the central directory."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.session, self.position = url, session, 0
        response = session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["content-length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def readinto(self, buffer) -> int:
        end = min(self.position + len(buffer), self.size)
        if end <= self.position:
            return 0
        data = fetch_range(self.session, self.url, self.position, end - 1)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def fetch_range(session: requests.Session, url: str, first: int, last: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=60)
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"{url} ignored the range request (status {response.status_code})")
    return response.content


class RemoteZip:
    """Reads single members of a remote zip archive with one range request each."""

    def __init__(self, url: str):
        self.url = url
        self.local = threading.local()
        with io.BufferedReader(HttpRangeFile(url, self.session()), buffer_size=1 << 20) as remote:
            self.members = {Path(info.filename).name: info for info in zipfile.ZipFile(remote).infolist()}

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def read(self, name: str) -> bytes:
        info = self.members[name]
        start = info.header_offset
        raw = fetch_range(self.session(), self.url, start, start + LOCAL_HEADER.size + HEADER_SLACK + info.compress_size - 1)
        signature, *_, name_length, extra_length = LOCAL_HEADER.unpack_from(raw)
        if signature != b"PK\x03\x04":
            raise RuntimeError(f"{name}: bad local header in {self.url}")
        data_start = LOCAL_HEADER.size + name_length + extra_length
        if data_start + info.compress_size > len(raw):
            raw = fetch_range(self.session(), self.url, start, start + data_start + info.compress_size - 1)
        data = raw[data_start : data_start + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(data)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{name}: unsupported compression {info.compress_type}")
        if zlib.crc32(data) != info.CRC:
            raise RuntimeError(f"{name}: checksum mismatch")
        return data


def fetch_members(archive_url: str, ids: list[str], workers: int) -> list[str]:
    """Download the given image IDs from one remote zip; returns the IDs that failed."""
    archive = RemoteZip(archive_url)

    def fetch(image_id: str) -> None:
        target = image_path("disc21", image_id)
        for attempt in range(RETRIES):
            try:
                data = archive.read(target.name)
                break
            except (requests.RequestException, RuntimeError):
                if attempt == RETRIES - 1:
                    raise
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, image_id): image_id for image_id in ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc=Path(archive_url).name, file=sys.stderr):
            if future.exception() is not None:
                print(f"failed {futures[future]}: {future.exception()}", file=sys.stderr)
                failed.append(futures[future])
    return failed


def caltech101(args: argparse.Namespace) -> list[str]:
    if not CALTECH_ROOT.exists():
        archive = download_file(CALTECH_URL, CALTECH_ROOT.parent / "caltech-101.zip")
        with zipfile.ZipFile(archive) as outer:
            inner = next(name for name in outer.namelist() if name.endswith("101_ObjectCategories.tar.gz"))
            with outer.open(inner) as tar_stream, tarfile.open(fileobj=tar_stream, mode="r|gz") as tar:
                tar.extractall(CALTECH_ROOT.parent, filter="data")
        archive.unlink()
    if not splits_exist("caltech101"):
        write_splits("caltech101", make_caltech_splits(caltech_image_ids()))
    return []


def disc21(args: argparse.Namespace) -> list[str]:
    download_file(f"{DISC21_URL}/dev_ground_truth.csv", DISC21_DIR / "dev_ground_truth.csv")
    if args.attributions:
        download_file(f"{DISC21_URL}/disc21_yfcc_attributions.csv", DISC21_DIR / "disc21_yfcc_attributions.csv")
    if not splits_exist("disc21"):
        write_splits("disc21", make_disc21_splits(dev_ground_truth(), args.index_size, args.negative_queries))

    wanted = sorted({image_id for name in ("index", "val_distractors", "test_distractors", "dev_calibration", "dev_report")
                     for image_id in load_split("disc21", name)})
    if args.limit:
        wanted = [i for i in wanted if i.startswith("R")][: args.limit] + [i for i in wanted if i.startswith("Q")][: args.limit]
    missing = [i for i in wanted if not image_path("disc21", i).exists()]
    print(f"{len(wanted) - len(missing):,} of {len(wanted):,} DISC21 images present, fetching {len(missing):,}", file=sys.stderr)
    for folder in ("references", "queries"):
        (DISC21_DIR / folder).mkdir(parents=True, exist_ok=True)

    by_archive: dict[str, list[str]] = {}
    for image_id in missing:
        archive = "dev_queries.zip" if image_id.startswith("Q") else f"references_{int(image_id[1:]) // DISC21_IMAGES_PER_REFERENCE_ZIP}.zip"
        by_archive.setdefault(archive, []).append(image_id)
    failed = []
    for archive, ids in sorted(by_archive.items()):
        failed += fetch_members(f"{DISC21_URL}/{archive}", ids, args.workers)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("caltech101", "disc21"))
    parser.add_argument("--index-size", type=int, default=50_000, help="DISC21 references in the index")
    parser.add_argument("--negative-queries", type=int, default=10_000, help="DISC21 development queries without a match")
    parser.add_argument("--no-attributions", dest="attributions", action="store_false", help="skip the 187 MB attribution CSV")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="fetch only the first N references and N queries (smoke test)")
    args = parser.parse_args()

    failed = {"caltech101": caltech101, "disc21": disc21}[args.dataset](args)
    if failed:
        print(f"{len(failed)} images failed; rerun to retry them", file=sys.stderr)
        sys.exit(1)
    print(f"{args.dataset} ready")


if __name__ == "__main__":
    main()
