"""Small local catalog with atomic stage checkpoints and a single-writer lock."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .storage import save_json


async def local_io(function, *args, **kwargs):
    """Finish an in-flight file operation before releasing the directory lock."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class LocalStore:
    def __init__(self, output_dir):
        self.directory = Path(output_dir)
        self.path = self.directory / "catalog.json"

    @contextmanager
    def writer(self):
        """Serialize directory changes; fail clearly if another process owns it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".raft.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt

                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)

                def acquire():
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

                def release():
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release():
                    fcntl.flock(lock, fcntl.LOCK_UN)

            try:
                acquire()
            except OSError as exc:
                raise RuntimeError("Local index is busy in another pipeline/process") from exc
            try:
                yield
            finally:
                release()

    def read(self):
        if not self.path.exists():
            return {"format": 1, "revision": 0, "cases": {}, "extraction_failures": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("format") != 1:
            raise ValueError("Unsupported local catalog format")
        return value

    def save(self, catalog):
        revision = catalog["revision"] + 1
        save_json(self.path, {**catalog, "revision": revision})
        catalog["revision"] = revision

    def cache_dir(self, catalog):
        return self.directory / "indexes" / str(catalog["revision"])
