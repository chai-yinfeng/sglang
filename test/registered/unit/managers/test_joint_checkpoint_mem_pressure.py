"""Joint checkpoint memory-pressure valve, CPU only, on the fixtures of
test_joint_checkpoint_truncate.py (Scheduler.__new__ over a real
UnifiedRadixCache, real pools and real Req objects), in a 64-token pool.

Pins and truncate holds must never make the scheduler fail an allocation or
retract / abort a request: when the pool is locked full,

- an allocation (alloc_token_slots, the prefill / decode KV allocation)
  degrades pins, then truncate holds, until it fits instead of raising OOM;
- a decode step that does not fit degrades them before retract_decode, which
  runs only once every pin and hold is gone;
- a waiting request refused for lack of tokens is admitted once enough pins
  are degraded, or, if it cannot fit even then, leaves every pin as it was.

Pins go in --jc-pin-overflow order (lru: least recently touched first;
next-use: expected to be used latest first), truncate holds last. Without a
shortfall nothing is degraded. Degraded pins are counted apart from the pin
cap's (jc_pins_mem_degraded_total, jc_holds_mem_dropped_total).
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_joint_checkpoint_truncate as T  # noqa: E402

from sglang.srt.managers.io_struct import PinReqInput, UnpinReqInput  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.managers.scheduler import JcPin  # noqa: E402
from sglang.srt.mem_cache.allocation import alloc_token_slots  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

OUTPUT = T.OUTPUT
SESSION = len(T.FILL)  # 16 tokens: a 10-token prompt and 6 outputs


def disjoint(k):
    return [p + 1000 * (k + 1) for p in T.PROMPT]


class _MemPressureFixture(T._PinOnFinishFixture):
    pool_size = 64

    def make_scheduler(self, *args, max_ratio=1.0, **kw):
        s = super().make_scheduler(*args, max_ratio=max_ratio, **kw)
        s.init_jc_memory_pressure_valve()
        return s

    def pinned(self, s, rid, k, t, resume_in_s=None):
        """Session k (16 tokens) finished at time t with a pin-on-finish hold."""
        self.now = t
        req = self.armed(s, rid, prompt=disjoint(k))
        self.finish(req)
        s.running_batch.filter_batch()
        self.assertIn(rid, s.pin_table)
        if resume_in_s is not None:
            out = s.pin_request(PinReqInput(rid=rid, resume_in_s=resume_in_s))
            self.assertTrue(out.success, out.message)
        return req

    def held(self, s, k, t, pos=13):
        """A running session k truncated at pos at time t: a truncate hold of
        pos tokens for its successor "succ"."""
        self.now = t
        old = self.start("old", disjoint(k), OUTPUT)
        s.running_batch.reqs = s.running_batch.reqs + [old]
        succ, err = self.truncate(s, "old", pos, prefix=(disjoint(k) + OUTPUT)[:pos])
        self.assertIsNone(err)
        s.running_batch.filter_batch()
        self.assertEqual(s.pin_table["succ"].successor_of, "old")
        return succ

    def running(self, s, rid, prompt, n_out):
        req = self.start(rid, prompt, list(range(500, 500 + n_out)))
        s.running_batch.reqs = s.running_batch.reqs + [req]
        return req

    def assert_full(self):
        self.assertEqual(self.allocator.available_size(), 0)
        self.assertEqual(self.cache.evictable_size(), 0)

    def assert_mem_degraded(self, s, rids):
        for rid in rids:
            self.assertNotIn(rid, s.pin_table)
            self.assertTrue(s.jc_degraded_pins[rid].degraded_by_memory)
        self.assertEqual(s.jc_pins_mem_degraded_total, len(rids))
        self.assertEqual(s.jc_pins_degraded_total, 0)


class TestJointCheckpointMemPressureAlloc(_MemPressureFixture):
    """An allocation the free and evictable tokens cannot serve."""

    def locked_full(self, overflow, hints=(None, None)):
        """Pins a (t=1000) and b (t=1001), a truncate hold of 13 tokens taken
        earlier (t=999), and a running request on the remaining 19 tokens."""
        s = self.make_scheduler(overflow=overflow)
        succ = self.held(s, 2, 999.0)
        self.pinned(s, "a", 0, 1000.0, resume_in_s=hints[0])
        self.pinned(s, "b", 1, 1001.0, resume_in_s=hints[1])
        self.running(s, "r", disjoint(3), 64 - 2 * SESSION - 13 - 10)
        self.assert_full()
        return s, succ

    def test_prefill_allocation_degrades_a_pin_instead_of_oom(self):
        s, _ = self.locked_full("lru")
        out = alloc_token_slots(self.cache, 10)
        self.assertEqual(len(out), 10)
        # Least recently touched pin first; the (older) hold is left alone.
        self.assert_mem_degraded(s, ["a"])
        self.assertEqual(set(s.pin_table), {"b", "succ"})
        self.assertEqual(s.jc_holds_mem_dropped_total, 0)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertIn("degraded under memory pressure", out.message)

    def test_next_use_degrades_the_pin_used_latest(self):
        # a is used again at 1010, b at 1301: LRU would take a.
        s, _ = self.locked_full("next-use", hints=(10.0, 300.0))
        alloc_token_slots(self.cache, 10)
        self.assert_mem_degraded(s, ["b"])
        self.assertEqual(set(s.pin_table), {"a", "succ"})

    def test_truncate_holds_go_after_every_pin(self):
        s, succ = self.locked_full("lru")
        out = alloc_token_slots(self.cache, 40)  # 2 x 16 pinned is not enough
        self.assertEqual(len(out), 40)
        self.assert_mem_degraded(s, ["a", "b"])
        self.assertEqual(s.pin_table, {})
        self.assertEqual(s.jc_holds_mem_dropped_total, 1)
        # The successor finds no hold to drop at admission.
        s._jc_release_truncate_hold(succ, "admitted")
        self.assertEqual(s.pin_table, {})

    def test_still_short_with_everything_degraded_raises_as_before(self):
        s, _ = self.locked_full("lru")
        with self.assertRaises(RuntimeError):
            alloc_token_slots(self.cache, 64)
        self.assertEqual(s.pin_table, {})
        self.assertEqual(s.jc_pins_mem_degraded_total, 2)
        self.assertEqual(s.jc_holds_mem_dropped_total, 1)

    def test_no_shortfall_degrades_nothing(self):
        s = self.make_scheduler(overflow="lru")
        self.held(s, 2, 999.0)
        self.pinned(s, "a", 0, 1000.0)
        self.pinned(s, "b", 1, 1001.0)
        free = self.allocator.available_size()
        out = alloc_token_slots(self.cache, free)
        self.assertEqual(len(out), free)
        self.assertEqual(set(s.pin_table), {"a", "b", "succ"})
        self.assertEqual(s.jc_degraded_pins, {})
        self.assertEqual(s.jc_pins_mem_degraded_total, 0)
        self.assertEqual(s.jc_holds_mem_dropped_total, 0)

    def test_an_unpinned_prefix_is_evicted_before_any_pin(self):
        s = self.make_scheduler(overflow="lru")
        self.pinned(s, "a", 0, 1000.0)
        unpinned = self.start("u", disjoint(1), OUTPUT)
        self.finish(unpinned)  # in the tree, evictable
        self.running(s, "r", disjoint(3), 64 - 2 * SESSION - 10)
        self.assertEqual(self.allocator.available_size(), 0)
        alloc_token_slots(self.cache, SESSION)
        self.assertEqual(set(s.pin_table), {"a"})
        self.assertEqual(s.jc_pins_mem_degraded_total, 0)

    def test_flag_off_installs_no_hook(self):
        s = self.make_scheduler(enable=False)
        self.assertIsNone(getattr(self.cache, "jc_memory_pressure_hook", None))
        self.assertFalse(s.enable_joint_checkpoint)


class TestJointCheckpointMemPressureAllocPythonCore(
    TestJointCheckpointMemPressureAlloc
):
    tree_core_backend = "python"


class TestJointCheckpointMemPressureDecode(_MemPressureFixture):
    """update_running_batch when the next decode step does not fit."""

    def decode(self, s, retract=None):
        s.enable_skip_finishing_decode = False
        s.new_token_ratio_tracker = SimpleNamespace(current=0.5, decay_step=MagicMock())
        s.decode_offload_manager = None
        s.metrics_reporter = SimpleNamespace(num_retracted_reqs=0, enable_metrics=False)
        batch = s.running_batch
        batch.tree_cache = self.cache
        batch.token_to_kv_pool_allocator = self.allocator
        batch.spec_algorithm = s.spec_algorithm
        retract = retract or MagicMock(return_value=([], 0.5, []))
        with (
            patch.object(ScheduleBatch, "retract_decode", retract),
            patch.object(ScheduleBatch, "prepare_for_decode") as prepare,
        ):
            out = s.update_running_batch(batch)
        return out, retract, prepare

    def test_decode_degrades_a_pin_instead_of_retracting(self):
        s = self.make_scheduler(overflow="next-use")
        self.pinned(s, "a", 0, 1000.0, resume_in_s=10.0)
        self.pinned(s, "b", 1, 1001.0, resume_in_s=300.0)
        r = self.running(s, "r", disjoint(2), 64 - 2 * SESSION - 10)
        self.assert_full()
        out, retract, prepare = self.decode(s)
        retract.assert_not_called()
        prepare.assert_called_once()
        self.assertEqual(out.reqs, [r])
        self.assert_mem_degraded(s, ["b"])
        self.assertEqual(set(s.pin_table), {"a"})

    def test_decode_lru_order(self):
        s = self.make_scheduler(overflow="lru")
        self.pinned(s, "a", 0, 1000.0, resume_in_s=10.0)
        self.pinned(s, "b", 1, 1001.0, resume_in_s=300.0)
        self.running(s, "r", disjoint(2), 64 - 2 * SESSION - 10)
        _, retract, _ = self.decode(s)
        retract.assert_not_called()
        self.assert_mem_degraded(s, ["a"])

    def test_retract_only_after_every_pin_and_hold_is_degraded(self):
        s = self.make_scheduler(overflow="lru")
        a = self.pinned(s, "a", 0, 1000.0)
        # A running request continues a's path: dropping a's pin frees nothing.
        prompt = list(a.full_untruncated_fill_ids) + list(range(700, 720))
        r = self.running(s, "r", prompt, 64 - SESSION - 20)
        self.assertEqual(r.prefix_len, SESSION)
        s.pin_table["succ"] = JcPin(
            lock=self.cache.lock(r.last_node),
            pinned_len=SESSION,
            pinned_at=900.0,  # older than a: still dropped after it
            expires_at=1030.0,
            successor_of="x",
        )
        self.assert_full()
        order = []
        real_check = ScheduleBatch.check_decode_mem

        def check(batch, *args, **kw):
            order.append(set(s.pin_table))
            return real_check(batch, *args, **kw)

        with patch.object(ScheduleBatch, "check_decode_mem", check):
            _, retract, _ = self.decode(s)
        retract.assert_called_once()
        self.assertEqual(order, [{"a", "succ"}, {"succ"}, set()])
        self.assertEqual(s.pin_table, {})
        self.assertEqual(s.jc_pins_mem_degraded_total, 1)
        self.assertEqual(s.jc_holds_mem_dropped_total, 1)

    def test_decode_that_fits_degrades_nothing(self):
        s = self.make_scheduler(overflow="lru")
        self.pinned(s, "a", 0, 1000.0)
        self.running(s, "r", disjoint(2), 4)
        _, retract, prepare = self.decode(s)
        retract.assert_not_called()
        prepare.assert_called_once()
        s.new_token_ratio_tracker.decay_step.assert_called_once()
        self.assertEqual(set(s.pin_table), {"a"})
        self.assertEqual(s.jc_pins_mem_degraded_total, 0)


class _LiveAdder:
    """Admits a request when ``need`` tokens are free or evictable, as
    PrefillAdder.add_one_req judges the live tree."""

    def __init__(self, cache, need):
        self.cache = cache
        self.need = need
        self.can_run_list = []
        self.calls = 0

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.calls += 1
        allocator = self.cache.token_to_kv_pool_allocator
        if allocator.available_size() + self.cache.evictable_size() < self.need:
            return AddReqResult.NO_TOKEN
        self.can_run_list.append(req)
        return AddReqResult.NO_TOKEN  # admitted, and the pool is now spent


class TestJointCheckpointMemPressureAdmission(_MemPressureFixture):
    """_jc_admit_under_memory_pressure: a waiting request refused NO_TOKEN."""

    def locked_full(self):
        s = self.make_scheduler(overflow="lru")
        s.truncation_align_size = None
        succ = self.held(s, 2, 999.0)
        self.pinned(s, "a", 0, 1000.0)
        self.pinned(s, "b", 1, 1001.0)
        self.running(s, "r", disjoint(3), 64 - 2 * SESSION - 13 - 10)
        self.assert_full()
        return s, succ

    def test_admitted_once_enough_pins_are_degraded(self):
        s, _ = self.locked_full()
        req = T.make_req("w", disjoint(4))
        adder = _LiveAdder(self.cache, need=20)
        res = s._jc_admit_under_memory_pressure(adder, req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(res, AddReqResult.NO_TOKEN)  # the next candidate's verdict
        self.assert_mem_degraded(s, ["a", "b"])
        self.assertEqual(set(s.pin_table), {"succ"})

    def test_successor_keeps_its_own_hold(self):
        s, succ = self.locked_full()
        adder = _LiveAdder(self.cache, need=40)  # needs the hold's 13 too
        s._jc_admit_under_memory_pressure(adder, succ)
        self.assertEqual(adder.can_run_list, [])
        # Not admissible without dropping its own hold: nothing changed.
        self.assertEqual(set(s.pin_table), {"a", "b", "succ"})
        self.assertEqual(s.jc_degraded_pins, {})
        self.assertEqual(s.jc_pins_mem_degraded_total, 0)
        self.assert_full()

    def test_infeasible_request_leaves_every_pin_locked(self):
        s, _ = self.locked_full()
        req = T.make_req("w", disjoint(4))
        adder = _LiveAdder(self.cache, need=60)
        res = s._jc_admit_under_memory_pressure(adder, req)
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.calls, 3)
        self.assertEqual(set(s.pin_table), {"a", "b", "succ"})
        self.assertEqual(s.jc_degraded_pins, {})
        self.assertEqual(
            (s.jc_pins_mem_degraded_total, s.jc_holds_mem_dropped_total), (0, 0)
        )
        self.assert_full()
        # The restored locks are real: an allocation still has to degrade them.
        alloc_token_slots(self.cache, 10)
        self.assert_mem_degraded(s, ["a"])

    def test_request_with_a_host_hit_goes_through_the_valve_too(self):
        s, _ = self.locked_full()
        req = T.make_req("w", disjoint(4))
        req.host_hit_length = 4  # its load-back is budgeted by the adder
        adder = _LiveAdder(self.cache, need=20)
        s._jc_admit_under_memory_pressure(adder, req)
        self.assertEqual(adder.can_run_list, [req])
        self.assert_mem_degraded(s, ["a", "b"])
        self.assertEqual(set(s.pin_table), {"succ"})


if __name__ == "__main__":
    unittest.main()
