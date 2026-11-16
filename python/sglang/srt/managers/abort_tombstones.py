"""Aborts that arrived before their request was registered (joint checkpoint).

/generate and /abort_request are independent HTTP requests. With a single
tokenizer worker, TokenizerManager.abort_request drops an abort whose rid is not
in rid_to_state yet, so when the abort handler runs before the generate handler
registers the rid, the abort is lost and the request runs to completion.

A tombstone keeps such an abort (and its discard flag) for a short time. The
generate request consumes it when it registers the same rid (exact match; the
scheduler's prefix match is not reproduced here so that an abort of "a-1" never
hits "a-12"). Tombstones expire after ``ttl_s`` and the table holds at most
``max_entries``, evicting the oldest, so aborts for rids that never show up
cannot grow it without bound.
"""

import time
from collections import OrderedDict
from typing import Optional, Tuple


class AbortTombstones:
    TTL_S = 30.0
    MAX_ENTRIES = 4096

    def __init__(self, ttl_s: float = TTL_S, max_entries: int = MAX_ENTRIES):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        # rid -> (expiry, discard). Entries are only ever appended with the same
        # ttl, so insertion order is expiry order.
        self._entries: "OrderedDict[str, Tuple[float, bool]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, rid: str, discard: bool) -> None:
        now = time.monotonic()
        self._prune(now)
        old = self._entries.get(rid)
        if old is not None:
            # Repeated abort of a still-unregistered rid: keep one tombstone with
            # its original expiry; discard sticks if any of the aborts asked.
            self._entries[rid] = (old[0], old[1] or discard)
            return
        self._entries[rid] = (now + self.ttl_s, discard)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def pop(self, rid: str) -> Optional[bool]:
        """Consume rid's live tombstone; return its discard flag, or None."""
        if not self._entries:
            return None
        self._prune(time.monotonic())
        entry = self._entries.pop(rid, None)
        return None if entry is None else entry[1]

    def _prune(self, now: float) -> None:
        while self._entries:
            expiry, _ = next(iter(self._entries.values()))
            if expiry > now:
                break
            self._entries.popitem(last=False)
