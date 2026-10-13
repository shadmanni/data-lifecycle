"""
Pure-Python content-defined chunking (Gear-hash based, FastCDC-inspired).
Supports both in-memory byte buffers and streaming file I/O to avoid RAM spikes on large datasets.
No external C dependencies -> works identically on Windows, macOS, Linux.
"""
import hashlib
import math
import random
from typing import List, Generator, Tuple

random.seed(42)
GEAR_TABLE = [random.getrandbits(32) for _ in range(256)]


def hash_chunk(data: bytes) -> str:
    """SHA-1 hash of a chunk's bytes -> used as the chunk's fingerprint."""
    return hashlib.sha1(data).hexdigest()


def hash_chunk_compact(data: bytes) -> int:
    """
    64-bit integer fingerprint of a chunk's bytes.
    Uses 8 bytes instead of 40-character hex string, reducing memory by 80%.
    """
    digest = hashlib.sha1(data).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def gear_chunks(data: bytes, avg_size: int = 8192, min_size: int = 2048, max_size: int = 16384) -> List[Tuple[int, int]]:
    """
    Split in-memory `data` into content-defined chunks.
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


def gear_chunks_stream(
    stream,
    avg_size: int = 8192,
    min_size: int = 2048,
    max_size: int = 16384,
    buffer_size: int = 65536,
    compact: bool = False,
) -> Generator[str, None, None]:
    """
    Stream-based Gear chunking without loading the full file or payload into memory.
    Reads fixed buffer_size blocks from `stream` (binary file or buffer).
    Yields chunk hash strings (or compact 64-bit int if compact=True).
    """
    bits = max(1, int(math.log2(avg_size)))
    mask = (1 << bits) - 1

    fp = 0
    current_chunk = bytearray()

    while True:
        buf = stream.read(buffer_size)
        if not buf:
            break

        for byte_val in buf:
            current_chunk.append(byte_val)
            fp = ((fp << 1) + GEAR_TABLE[byte_val]) & 0xFFFFFFFF
            chunk_len = len(current_chunk)

            if chunk_len >= min_size and (fp & mask) == 0:
                chunk_bytes = bytes(current_chunk)
                yield hash_chunk_compact(chunk_bytes) if compact else hash_chunk(chunk_bytes)
                current_chunk.clear()
                fp = 0
            elif chunk_len >= max_size:
                chunk_bytes = bytes(current_chunk)
                yield hash_chunk_compact(chunk_bytes) if compact else hash_chunk(chunk_bytes)
                current_chunk.clear()
                fp = 0

    if len(current_chunk) > 0:
        chunk_bytes = bytes(current_chunk)
        yield hash_chunk_compact(chunk_bytes) if compact else hash_chunk(chunk_bytes)


def chunk_file(filepath: str, avg_size: int = 8192, min_size: int = 2048, max_size: int = 16384, compact: bool = False) -> List[str]:
    """
    Reads a file using buffered streaming and returns a list of chunk hashes.
    Memory-bounded: memory overhead is bounded to max_size + buffer_size regardless of file size.
    """
    with open(filepath, "rb") as f:
        return list(gear_chunks_stream(f, avg_size=avg_size, min_size=min_size, max_size=max_size, compact=compact))


if __name__ == "__main__":
    # quick self-test
    test_data = b"hello world " * 5000
    boundaries = gear_chunks(test_data)
    hashes = [hash_chunk(test_data[s:e]) for (s, e) in boundaries]
    print(f"In-memory chunker produced {len(hashes)} chunks from {len(test_data)} bytes")

    import io
    stream = io.BytesIO(test_data)
    stream_hashes = list(gear_chunks_stream(stream))
    print(f"Stream chunker produced {len(stream_hashes)} chunks")
    assert hashes == stream_hashes, "Stream and in-memory chunking hashes match perfectly!"
    print("Self-test passed!")