"""Joint checkpoint pin pressure valve (--jc-pin-overflow lru): which pins to
degrade so a new hold fits under --jc-pin-max-ratio."""

from __future__ import annotations

from typing import AbstractSet, Callable, List, Optional, Sequence


def select_pins_to_degrade(
    *,
    candidates: Sequence[str],
    fits: Callable[[AbstractSet[str]], bool],
) -> Optional[List[str]]:
    """The pins to degrade, a subset of ``candidates`` (least recently touched
    first), so that ``fits(degraded)`` holds; None when even degrading all of
    them would not make room (then nothing is degraded).

    ``fits`` judges distinct tree tokens, so degrading a pin whose path the
    kept pins or the new hold share frees nothing: candidates are taken in
    order until the new hold fits, then any taken one that is not needed for
    the fit is spared, most recently touched first.
    """
    if not fits(frozenset(candidates)):
        return None
    picked: List[str] = []
    for rid in candidates:
        if fits(frozenset(picked)):
            break
        picked.append(rid)
    for rid in reversed(list(picked)):
        rest = frozenset(r for r in picked if r != rid)
        if fits(rest):
            picked.remove(rid)
    return picked
