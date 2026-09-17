"""Read an HDF5 file that lives inside a single-member .tar.zst WITHOUT
extracting it (disk-quota workaround). A forward-only zstd stream is exposed
as a seekable file object: forward seeks skip bytes, backward seeks restart the
decompressor from the beginning. Small reads (HDF5 metadata / chunk index
pages) are cached so repeated metadata access never triggers restarts.

Typical use: read all small columns once each, walk the chunk index of the
`pixels` dataset, then iterate pixel chunks in ascending file order (one pass).
"""
from __future__ import annotations

import io
import time

import zstandard


class ZstTarMemberFile(io.RawIOBase):
    def __init__(self, path, cache_max=1 << 20, read_size=16 << 20, verbose=True):
        super().__init__()
        self.path = path
        self.cache_max = cache_max
        self.read_size = read_size
        self.verbose = verbose
        self.cache: dict[tuple[int, int], bytes] = {}
        self.restarts = 0
        self.bytes_streamed = 0
        self._open()
        import tarfile
        # let tarfile parse the (possibly GNU base-256 / pax) header of the first member
        tf = tarfile.open(fileobj=self._reader, mode='r|')
        m = tf.next()
        self.member_name, self.size, self.base = m.name, m.size, m.offset_data
        self._reader.close(); self._f.close()
        self._open()
        self.pos = 0

    # ---- raw stream handling -------------------------------------------------
    def _open(self):
        self._f = open(self.path, 'rb')
        self._reader = zstandard.ZstdDecompressor().stream_reader(self._f, read_size=self.read_size)
        self._abs = 0

    def _skip_to(self, absoff):
        if absoff < self._abs:
            self.restarts += 1
            if self.verbose:
                print(f'[zsth5] backward seek {self._abs} -> {absoff}: restart #{self.restarts}', flush=True)
            self._reader.close(); self._f.close()
            self._open()
        while self._abs < absoff:
            n = min(self.read_size, absoff - self._abs)
            b = self._reader.read(n)
            if not b:
                raise EOFError('stream ended while skipping')
            self._abs += len(b)
            self.bytes_streamed += len(b)

    def _raw_read_abs(self, absoff, n):
        self._skip_to(absoff)
        out = bytearray()
        while len(out) < n:
            b = self._reader.read(n - len(out))
            if not b:
                break
            out += b
        self._abs += len(out)
        self.bytes_streamed += len(out)
        return bytes(out)

    # ---- file-like API (member-relative) ------------------------------------
    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = off
        elif whence == io.SEEK_CUR: self.pos += off
        elif whence == io.SEEK_END: self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        key = (self.pos, n)
        if n <= self.cache_max and key in self.cache:
            data = self.cache[key]
        else:
            data = self._raw_read_abs(self.base + self.pos, n)
            if n <= self.cache_max:
                self.cache[key] = data
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self):
        try:
            self._reader.close(); self._f.close()
        finally:
            super().close()


def open_h5_in_tar_zst(path, **kw):
    import h5py
    import hdf5plugin  # noqa: F401
    fobj = ZstTarMemberFile(path, **kw)
    f = h5py.File(fobj, 'r', rdcc_nbytes=0)
    return f, fobj


def iter_chunks_in_file_order(dset):
    """Yield (row_start, row_end) of every chunk of a 1-D-chunked dataset sorted by byte offset."""
    n = dset.id.get_num_chunks()
    infos = []
    for i in range(n):
        ci = dset.id.get_chunk_info(i)
        infos.append((ci.byte_offset, ci.chunk_offset[0]))
    infos.sort()
    rows_per = dset.chunks[0]
    for _, r0 in infos:
        yield r0, min(r0 + rows_per, dset.shape[0])
