"""Joint checkpoint pin policy for runtimes that know what they will use next
(the JC-SEM server preset), CPU only, on the fixtures of
test_joint_checkpoint_truncate.py (Scheduler.__new__ over a real
UnifiedRadixCache, real pools and real Req objects):

- Release (TestJointCheckpointRelease, Rust and Python tree cores):
  /unpin_request release=true drops the hold and frees right away the tail of
  the rid's recorded path that no lock holds and no other branch continues;
  without the flag the KV stays in the tree, evictable.
- Exclusive budget (TestJointCheckpointExclusiveBudget):
  --jc-pin-budget exclusive does not count the tree tokens in-flight requests
  hold themselves, and re-checks the cap once one of them leaves.
- Next-use valve (TestJointCheckpointNextUseValve): --jc-pin-overflow next-use
  degrades the pins expected to be used latest first (resume_in_s hint, else
  the expiry), and only pins expected later than the new hold.
"""

import os
import sys
import unittest
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_joint_checkpoint_truncate as T  # noqa: E402

from sglang.srt.managers.io_struct import PinReqInput, UnpinReqInput  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT, OUTPUT, FILL = T.PROMPT, T.OUTPUT, T.FILL


def disjoint(k, base=PROMPT):
    return [p + 1000 * (k + 1) for p in base]


class TestJointCheckpointRelease(T._PinOnFinishFixture):
    def held(self, s, rid, prompt=PROMPT, outputs=OUTPUT):
        req = self.armed(s, rid, prompt=prompt, outputs=outputs)
        self.finish(req)
        s.running_batch.filter_batch()
        return req

    def test_release_frees_the_finished_path_now(self):
        s = self.make_scheduler()
        self.held(s, "a")
        before = self.allocator.available_size()
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.released_tokens, len(FILL))
        self.assertEqual(s.jc_released_tokens_total, len(FILL))
        self.assertEqual(s.pin_table, {})
        self.assertEqual(self.match_len(FILL), 0)
        self.assertEqual(self.allocator.available_size(), before + len(FILL))
        self.assertEqual(self.allocator.available_size(), self.pool_size)

    def test_plain_unpin_leaves_the_kv_in_the_tree(self):
        s = self.make_scheduler()
        self.held(s, "a")
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertEqual(out.released_tokens, 0)
        self.assertEqual(self.match_len(FILL), len(FILL))

    def test_release_stops_where_another_branch_continues(self):
        s = self.make_scheduler()
        self.held(s, "a")
        other_out = [300, 301, 302]
        self.held(s, "b", outputs=other_out)  # shares PROMPT, then its own tail
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertEqual(out.released_tokens, len(OUTPUT))
        self.assertEqual(self.match_len(FILL), len(PROMPT))
        self.assertEqual(self.match_len(PROMPT + other_out), len(PROMPT) + 3)
        # b released too: now nothing continues the prompt, it goes as well.
        out = s.unpin_request(UnpinReqInput(rid="b", release=True))
        self.assertEqual(out.released_tokens, len(PROMPT) + 3)
        self.assertEqual(self.allocator.available_size(), self.pool_size)

    def test_release_keeps_what_a_running_request_holds(self):
        s = self.make_scheduler()
        self.held(s, "a")
        # A successor admitted on a's whole path (it appended after a's end):
        # its own lock covers a's path, and its extension continues it.
        succ = self.start("succ", FILL + [900, 901], [7])
        s.running_batch.reqs = [succ]
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertTrue(out.success)
        self.assertEqual(out.released_tokens, 0)
        self.assertEqual(self.match_len(FILL), len(FILL))

    def test_release_frees_only_the_unheld_tail(self):
        s = self.make_scheduler()
        self.held(s, "a")
        # Something else holds the prompt (e.g. a request admitted on it).
        node = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", PROMPT)))
        ).last_device_node
        other = self.cache.lock(node)
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertEqual(out.released_tokens, len(OUTPUT))
        self.assertEqual(self.match_len(FILL), len(PROMPT))
        self.cache.unlock(other)

    def test_release_of_a_degraded_hold_frees_what_is_left(self):
        s = self.make_scheduler(max_ratio=0.07, overflow="lru")  # limit 17
        self.now = 1000.0
        self.held(s, "a", prompt=disjoint(0))
        self.now = 1001.0
        b = self.held(s, "b", prompt=disjoint(1))
        self.assertEqual(b.jc_pin_outcome["jc_pin_degraded"], ["a"])
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertIn("degraded", out.message)
        self.assertEqual(out.released_tokens, len(FILL))
        self.assertEqual(self.match_len(disjoint(0) + OUTPUT), 0)
        self.assertEqual(self.match_len(disjoint(1) + OUTPUT), len(FILL))

    def test_release_of_an_unknown_rid_is_a_noop(self):
        s = self.make_scheduler()
        out = s.unpin_request(UnpinReqInput(rid="nobody", release=True))
        self.assertTrue(out.success)
        self.assertEqual(out.released_tokens, 0)


class TestJointCheckpointReleasePythonCore(TestJointCheckpointRelease):
    tree_core_backend = "python"


class TestJointCheckpointExclusiveBudget(T._SharedPrefixFixture):
    def make_scheduler(self, *args, budget="exclusive", **kw):
        s = super().make_scheduler(*args, **kw)
        s.jc_pin_budget = budget
        return s

    def test_tokens_a_running_request_holds_do_not_count(self):
        s = self.make_scheduler()
        self.hold(s, 0)
        s.running_batch.filter_batch()
        self.assertEqual(s._jc_pinned_tokens_total(), 8000)
        r1 = self.armed(s, "r1", prompt=T.session_prompt(1), outputs=[7])
        self.assertEqual(r1.prefix_len, T.PREFIX_LEN)
        # r1 runs on the shared prefix: only r0's own tail is kept by pins alone.
        self.assertEqual(s._jc_pinned_tokens_total(), T.TAIL_LEN)
        s.jc_pin_budget = "distinct"
        self.assertEqual(s._jc_pinned_tokens_total(), 8000)

    def test_a_hold_fits_because_its_prefix_is_held_by_a_running_request(self):
        # limit 9830: two finished sessions are 4300 + 2 x 3700 = 11700 distinct.
        s = self.make_scheduler(max_ratio=0.3, overflow="lru")
        self.hold(s, 0)
        s.running_batch.filter_batch()
        r2 = self.start("r2", T.session_prompt(2), [7])  # runs, takes no hold
        s.running_batch.reqs = s.running_batch.reqs + [r2]
        r1 = self.hold(s, 1)
        self.assertEqual(r1.jc_pin_outcome, {"jc_pinned_len": 8000})
        self.assertEqual(set(s.pin_table), {"r0", "r1"})
        self.assertEqual(s._jc_pinned_tokens_total(), 2 * T.TAIL_LEN)
        # Nothing changed in flight: no re-check, nothing degraded.
        s._jc_on_step(s.running_batch)
        self.assertEqual(s.jc_pins_degraded_total, 0)
        # r2 leaves: the prefix is now kept by pins alone, 11700 > 9830, so
        # the next step degrades the least recently touched pin.
        self.finish(r2)
        s.running_batch.filter_batch()
        s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))
        self.assertEqual(s.jc_pins_degraded_total, 1)
        self.assertEqual(set(s.pin_table), {"r1"})
        self.assertLessEqual(s._jc_pinned_tokens_total(), int(0.3 * self.pool_size))

    def test_distinct_budget_refuses_the_same_hold(self):
        s = self.make_scheduler(max_ratio=0.3, overflow="refuse", budget="distinct")
        self.hold(s, 0)
        s.running_batch.filter_batch()
        self.armed(s, "r2", prompt=T.session_prompt(2), outputs=[7])
        r1 = self.hold(s, 1)
        self.assertEqual(r1.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertIn("pin cap exceeded", r1.jc_pin_outcome["jc_pin_message"])


class TestJointCheckpointNextUseValve(T._PinOnFinishFixture):
    """Disjoint 16-token holds under a 40-token limit (int(0.16 x 256))."""

    def held_at(self, s, rid, k, t, resume_in_s=None):
        self.now = t
        req = self.armed(s, rid, prompt=disjoint(k))
        self.finish(req)
        s.running_batch.filter_batch()
        if resume_in_s is not None:
            out = s.pin_request(PinReqInput(rid=rid, resume_in_s=resume_in_s))
            self.assertTrue(out.success, out.message)
        return req

    def running(self, s, rid, k):
        req = self.start(rid, disjoint(k), OUTPUT)
        s.running_batch.reqs = s.running_batch.reqs + [req]
        return req

    def test_degrades_the_pin_used_latest(self):
        s = self.make_scheduler(max_ratio=0.16, overflow="next-use")
        self.held_at(s, "a", 0, 1000.0, resume_in_s=10.0)  # used at 1010
        self.held_at(s, "b", 1, 1001.0, resume_in_s=300.0)  # used at 1301
        self.now = 1002.0
        self.running(s, "c", 2)
        out = s.pin_request(PinReqInput(rid="c", resume_in_s=20.0))  # at 1022
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.degraded, ["b"])  # LRU would have picked a
        self.assertEqual(set(s.pin_table), {"a", "c"})

    def test_a_hold_used_after_every_other_degrades_nothing(self):
        s = self.make_scheduler(max_ratio=0.16, overflow="next-use")
        self.held_at(s, "a", 0, 1000.0, resume_in_s=10.0)
        self.held_at(s, "b", 1, 1001.0, resume_in_s=30.0)
        self.now = 1002.0
        self.running(s, "c", 2)
        out = s.pin_request(PinReqInput(rid="c", resume_in_s=500.0))
        # Partial hold of what fits beside a and b (40 - 32 = 8 tokens).
        self.assertTrue(out.success, out.message)
        self.assertEqual((out.degraded, out.pinned_len), ([], 8))
        self.assertIn("partial hold", out.message)
        self.assertEqual(s.jc_pins_degraded_total, 0)

    def test_no_room_and_no_later_pin_is_refused(self):
        s = self.make_scheduler(max_ratio=0.125, overflow="next-use")  # limit 32
        self.held_at(s, "a", 0, 1000.0, resume_in_s=10.0)
        self.held_at(s, "b", 1, 1001.0, resume_in_s=30.0)
        self.now = 1002.0
        c = self.armed(s, "c", prompt=disjoint(2))
        self.finish(c)  # finish hold, no hint: expected at its expiry (1602)
        self.assertEqual(c.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertIn(
            "degrading every pin expected to be used later would not make room",
            c.jc_pin_outcome["jc_pin_message"],
        )
        self.assertEqual(set(s.pin_table), {"a", "b"})

    def test_without_hints_the_expiry_orders_the_pins(self):
        s = self.make_scheduler(max_ratio=0.16, overflow="next-use")
        self.now = 1000.0
        a = self.armed(s, "a", prompt=disjoint(0), ttl_s=500.0)  # expires 1500
        self.finish(a)
        self.now = 1001.0
        b = self.armed(s, "b", prompt=disjoint(1), ttl_s=60.0)  # expires 1061
        self.finish(b)
        self.now = 1002.0
        c = self.armed(s, "c", prompt=disjoint(2), ttl_s=100.0)  # expires 1102
        self.finish(c)
        self.assertEqual(c.jc_pin_outcome["jc_pin_degraded"], ["a"])

    def test_a_renewal_moves_the_hint(self):
        s = self.make_scheduler(max_ratio=0.16, overflow="next-use")
        self.held_at(s, "a", 0, 1000.0, resume_in_s=300.0)
        self.held_at(s, "b", 1, 1001.0, resume_in_s=200.0)
        self.now = 1002.0
        self.assertTrue(s.pin_request(PinReqInput(rid="a", resume_in_s=5.0)).success)
        self.assertEqual(s.pin_table["a"].expected_use_at, 1007.0)
        self.running(s, "c", 2)
        out = s.pin_request(PinReqInput(rid="c", resume_in_s=20.0))
        self.assertEqual(out.degraded, ["b"])

    def test_negative_hint_is_refused(self):
        s = self.make_scheduler(overflow="next-use")
        out = s.pin_request(PinReqInput(rid="x", resume_in_s=-1.0))
        self.assertFalse(out.success)
        self.assertIn("resume_in_s", out.message)


if __name__ == "__main__":
    unittest.main()
