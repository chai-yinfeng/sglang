"""Joint checkpoint on hybrid linear-attention models (GDN / mamba layers).

A recurrent state can only return to positions where it was saved. Each
request keeps the states the decode track buffer is about to overwrite in a
small ring (--jc-mamba-history-slots), and a pin or truncate lands on the
nearest saved state at or before the position it asks for. Design notes:
docs/design-hybrid-truncate.md in the continuous-inference project.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, KVCache

logger = logging.getLogger(__name__)


class JcMambaSnapshot(msgspec.Struct, frozen=True):
    """A mamba pool slot holding a request's recurrent state after ``seqlen``
    tokens. ``order`` is a global push counter (oldest is smallest)."""

    seqlen: int
    # Shape [1], a slot id of the mamba pool; the history owns it.
    slot: Any
    order: int


class JcMambaState(msgspec.Struct, frozen=True):
    """A state a running request can return to.

    source: "tree" (a mamba value on a node of its tree path; ``node``),
    "ring" (its snapshot history), "track" (an entry of its decode track
    buffer; ``track_idx``), or "copy" (its working slot, valid only when no
    forward is in flight for it).
    """

    seqlen: int
    source: str
    node: Any = None
    track_idx: Optional[int] = None


class JcMambaCapture(msgspec.Struct, frozen=True):
    """A pin registered for a position decode has not reached yet."""

    target: int
    ttl_s: float


class JcMambaHistory:
    """Per-request rings of at most ``capacity`` snapshots, keyed by the
    request's req_to_token row. With capacity 0 nothing is kept, but the track
    buffer bookkeeping (retire_track_slots) still runs."""

    def __init__(self, *, capacity: int, allocator: MambaSlotAllocator):
        self.capacity = capacity
        self.allocator = allocator
        self.rings: Dict[int, List[JcMambaSnapshot]] = {}
        self.num_slots = 0
        # Pushes that found no free mamba slot and were dropped.
        self.alloc_failures = 0
        self._order = 0

    def retire_track_slots(
        self,
        *,
        reqs: Sequence[Req],
        track_mask: Sequence[bool],
        pool: HybridReqToTokenPool,
    ) -> None:
        """Before a decode step whose forward writes a track state for the
        masked requests: the track entry it writes (mamba_next_track_idx)
        stops holding the state it names, so that name is dropped, and with
        capacity the old state moves into the ring behind a fresh slot. No
        free slot: the state is overwritten as without the ring."""
        for req, tracked in zip(reqs, track_mask):
            kv = req.kv
            if not tracked or kv.mamba_ping_pong_track_buffer is None:
                continue
            idx = kv.mamba_next_track_idx
            if idx == kv.mamba_last_track_idx:
                seqlen, kv.mamba_last_track_seqlen = kv.mamba_last_track_seqlen, None
            else:
                seqlen, kv.mamba_prev_track_seqlen = kv.mamba_prev_track_seqlen, None
            if seqlen is None or self.capacity <= 0:
                continue
            new_slot = self.allocator.alloc(1)
            if new_slot is None:
                self.alloc_failures += 1
                continue
            old_slot = swap_track_slot(pool=pool, req=req, idx=idx, new_slot=new_slot)
            self._push(kv.req_pool_idx, seqlen, old_slot)

    def _push(self, row: int, seqlen: int, slot: torch.Tensor) -> None:
        ring = self.rings.setdefault(row, [])
        ring.append(JcMambaSnapshot(seqlen=seqlen, slot=slot, order=self._order))
        self._order += 1
        self.num_slots += 1
        while len(ring) > self.capacity:
            self._free([ring.pop(0)])

    def snapshots(self, row: Optional[int]) -> List[JcMambaSnapshot]:
        return list(self.rings.get(row, ()))

    def take(self, row: int, seqlen: int) -> Optional[torch.Tensor]:
        """Remove the snapshot at ``seqlen`` from the row's ring and hand its
        slot to the caller."""
        ring = self.rings.get(row, [])
        for i, snap in enumerate(ring):
            if snap.seqlen == seqlen:
                del ring[i]
                self.num_slots -= 1
                return snap.slot
        return None

    def release_row(self, row: Optional[int]) -> None:
        ring = self.rings.pop(row, None)
        if ring:
            self._free(ring)

    def reclaim(self, num_slots: int) -> int:
        """Free up to ``num_slots`` snapshots, oldest first across requests;
        returns how many were freed."""
        if num_slots <= 0 or self.num_slots == 0:
            return 0
        oldest = sorted(
            ((snap.order, row) for row, ring in self.rings.items() for snap in ring)
        )[:num_slots]
        victims = []
        for order, row in oldest:
            ring = self.rings[row]
            (i,) = [i for i, snap in enumerate(ring) if snap.order == order]
            victims.append(ring.pop(i))
        self._free(victims)
        return len(victims)

    def forget_all(self) -> None:
        """The pool's allocator was reset with every slot in it."""
        self.rings.clear()
        self.num_slots = 0

    def _free(self, snaps: Sequence[JcMambaSnapshot]) -> None:
        self.allocator.free(torch.cat([snap.slot for snap in snaps]))
        self.num_slots -= len(snaps)


def swap_track_slot(
    *,
    pool: HybridReqToTokenPool,
    req: Req,
    idx: int,
    new_slot: torch.Tensor,
) -> torch.Tensor:
    """Put ``new_slot`` (shape [1]) at track buffer entry ``idx`` and return
    the slot that was there (shape [1]); the caller owns it."""
    old_slot = req.kv.mamba_ping_pong_track_buffer[idx].unsqueeze(0).clone()
    pool.set_mamba_ping_pong_slot(req, idx, new_slot[0])
    return old_slot


def track_states(req: Req, *, track_buffer_size: int) -> List[JcMambaState]:
    """The states the request's decode track buffer names (the latest track
    and, with two entries, the one before it)."""
    kv = req.kv
    if kv.mamba_ping_pong_track_buffer is None:
        return []
    states = []
    if kv.mamba_last_track_seqlen is not None:
        states.append(
            JcMambaState(
                seqlen=kv.mamba_last_track_seqlen,
                source="track",
                track_idx=kv.mamba_last_track_idx,
            )
        )
    if kv.mamba_prev_track_seqlen is not None and track_buffer_size == 2:
        states.append(
            JcMambaState(
                seqlen=kv.mamba_prev_track_seqlen,
                source="track",
                track_idx=1 - kv.mamba_last_track_idx,
            )
        )
    return states


def select_mamba_states(
    states: Sequence[JcMambaState], target: int
) -> List[JcMambaState]:
    """The states at or before ``target``, nearest first; at equal positions
    a tree state (nothing to publish) comes first."""
    return sorted(
        (s for s in states if s.seqlen <= target),
        key=lambda s: (-s.seqlen, s.source != "tree"),
    )


def classify_pin_target(
    *, target: int, head: int, overlap: bool, head_tracked: bool
) -> str:
    """How a pin at ``target`` reaches its state; ``head`` is the position
    the latest launched forward of the request leaves its state at
    (kv_committed_len).

    "capture": decode has not produced the state yet; force a track at that
    step (or, under overlap, the in-flight step at ``head`` tracks anyway).
    "copy": the state at ``head`` sits in the working slot with no forward in
    flight (no overlap), so copying it is exact. "past": the nearest saved
    state at or before ``target``.
    """
    if target > head:
        return "capture"
    if target == head:
        if not overlap:
            return "copy"
        if head_tracked:
            return "capture"
    return "past"


def decode_track_masks(
    *,
    seq_lens: Sequence[int],
    capture_at: Sequence[Optional[int]],
    track_interval: int,
) -> Tuple[List[bool], List[bool]]:
    """Track masks of a decode step whose rows reach ``seq_lens``: the grid
    boundaries plus each row's pending capture point; the second mask is the
    same for the step after (overlap lookahead)."""
    mask = [n % track_interval == 0 or n == t for n, t in zip(seq_lens, capture_at)]
    mask_next = [
        (n + 1) % track_interval == 0 or n + 1 == t
        for n, t in zip(seq_lens, capture_at)
    ]
    return mask, mask_next


def snapshot_token_equivalent(*, snapshot_bytes: int, kv_bytes_per_token: float) -> int:
    """Full-attention KV tokens one mamba snapshot weighs in the pin cap."""
    if kv_bytes_per_token <= 0:
        return 0
    return math.ceil(snapshot_bytes / kv_bytes_per_token)


def mamba_slot_bytes(conv: Sequence[torch.Tensor], temporal: torch.Tensor) -> int:
    """Bytes of one slot of a mamba pool laid out [layers, slots, ...]."""
    return sum(t.numel() // t.shape[1] * t.element_size() for t in (*conv, temporal))


def hybrid_config_error(
    *,
    extra_buffer: bool,
    lazy: bool,
    int8_checkpoint: bool,
    replayssm: bool,
) -> Optional[str]:
    """Why --enable-joint-checkpoint cannot run on this hybrid configuration."""
    if not extra_buffer:
        return (
            "--enable-joint-checkpoint on a hybrid linear-attention model needs "
            "--mamba-radix-cache-strategy extra_buffer: without the extra buffer "
            "a running request has no saved recurrent state to pin or truncate to"
        )
    if lazy:
        return (
            "--enable-joint-checkpoint does not support "
            "--mamba-radix-cache-strategy extra_buffer_lazy; use extra_buffer"
        )
    if int8_checkpoint:
        return (
            "--enable-joint-checkpoint on a hybrid model does not support "
            "--enable-int8-mamba-checkpoint"
        )
    if replayssm:
        return (
            "--enable-joint-checkpoint on a hybrid model does not support "
            "--enable-linear-replayssm"
        )
    return None


class JcHybridSetup(msgspec.Struct, frozen=True):
    history: JcMambaHistory
    # Full-attention KV tokens one snapshot weighs in --jc-pin-max-ratio.
    snapshot_tokens: int


def setup_jc_hybrid(
    *,
    pool: HybridReqToTokenPool,
    kv_cache: KVCache,
    max_total_num_tokens: int,
    history_slots: int,
    extra_buffer: bool,
    lazy: bool,
    int8_checkpoint: bool,
    replayssm: bool,
    spec_enabled: bool,
) -> JcHybridSetup:
    """Validate a hybrid model's mamba cache for joint checkpoint, size one
    snapshot in KV tokens and build the snapshot rings. Raises ValueError on
    an unsupported configuration."""
    error = hybrid_config_error(
        extra_buffer=extra_buffer,
        lazy=lazy,
        int8_checkpoint=int8_checkpoint,
        replayssm=replayssm,
    )
    if error is not None:
        raise ValueError(error)
    if spec_enabled:
        logger.warning(
            "joint-checkpoint: speculative decoding on a hybrid model is not "
            "supported; pins and truncates do not account for its state rollback"
        )
    cache = pool.mamba_pool.mamba_cache
    snapshot_bytes = mamba_slot_bytes(cache.conv, cache.temporal)
    kv_bytes_per_token = sum(kv_cache.get_kv_size_bytes()) / kv_cache.size
    snapshot_tokens = snapshot_token_equivalent(
        snapshot_bytes=snapshot_bytes, kv_bytes_per_token=kv_bytes_per_token
    )
    logger.info(
        f"joint-checkpoint hybrid: KV pool {max_total_num_tokens} tokens "
        f"({kv_bytes_per_token:.0f} B/token), mamba pool {pool.mamba_pool.size} "
        f"slots, snapshot {snapshot_bytes} B = {snapshot_tokens} KV tokens, "
        f"history {history_slots} snapshots/request"
    )
    return JcHybridSetup(
        history=JcMambaHistory(capacity=history_slots, allocator=pool.mamba_allocator),
        snapshot_tokens=snapshot_tokens,
    )
