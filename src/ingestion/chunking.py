"""
Pure-Python content-defined chunking (Gear-hash based, FastCDC-inspired).
No external C dependencies -> works identically on Windows, macOS, Linux.
"""
import hashlib
import math
import random

random.seed(42)
GEAR_TABLE = [random.getrandbits(32) for _ in range(256)]


def gear_chunks(data: bytes, avg_size: int = 8192, min_size: int = 2048, max_size: int = 16384):
    """
    Split `data` into content-defined chunks.
    Returns a list of (start_offset, end_offset) tuples.
    """
    n = len(data)
    if n == 0:
        return []

    bits = max(1, int(math.log2(avg_size)))
    mask = (1 << bits) - 1

    chunks = []
    start = 0
    fp = 0

    for i in range(n):
        fp = ((fp << 1) + GEAR_TABLE[data[i]]) & 0xFFFFFFFF
        chunk_len = i - start + 1

        if chunk_len >= min_size and (fp & mask) == 0:
            chunks.append((start, i + 1))
            start = i + 1
            fp = 0
        elif chunk_len >= max_size:
            chunks.append((start, i + 1))
            start = i + 1
            fp = 0

    if start < n:
        chunks.append((start, n))

    return chunks


def hash_chunk(data: bytes) -> str:
    """SHA-1 hash of a chunk's bytes -> used as the chunk's fingerprint."""
    return hashlib.sha1(data).hexdigest()


def chunk_file(filepath: str, avg_size: int = 8192, min_size: int = 2048, max_size: int = 16384, max_bytes_to_read: int = 512_000):
    """
    Reads a file (up to max_bytes_to_read) and returns a list of chunk hashes.
    """
    with open(filepath, "rb") as f:
        data = f.read(max_bytes_to_read)
    boundaries = gear_chunks(data, avg_size, min_size, max_size)
    return [hash_chunk(data[s:e]) for (s, e) in boundaries]


if __name__ == "__main__":
    # quick self-test
    test_data = b"hello world " * 5000
    hashes = [hash_chunk(test_data[s:e]) for (s, e) in gear_chunks(test_data)]
    print(f"Test produced {len(hashes)} chunks from {len(test_data)} bytes")