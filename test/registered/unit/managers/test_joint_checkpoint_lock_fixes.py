"""Joint checkpoint lock fixes, CPU only, on the fixtures of
test_joint_checkpoint_truncate.py and test_joint_checkpoint_mem_pressure.py
(Scheduler.__new__ over a real UnifiedRadixCache, real pools and real Req
objects):

- Intake order (TestJointCheckpointIntakeOrder): a /generate carrying both
  truncate and jc_pin_on_finish is refused for a bad pin-on-finish before the
  truncate cuts the request it continues or takes a hold for it.
- Expired pins (TestJointCheckpointExpiredRelease): a pin whose TTL ran out
  is remembered, so a later /unpin_request release still frees its dead path
  and says it expired; the record is forgotten after --jc-pin-max-ttl-s, or
  oldest first past JC_EXPIRED_PINS_MAX.
- Memory-pressure valve (TestJointCheckpointMemPressureIdleDrops, Rust and
  Python tree cores): a pin whose lock frees nothing (its path is held by a
  running request) is kept once the shortfall is met; one that shares its path
  with a pin dropped after it is not.
- Lockable length (TestJointCheckpointFinishPinLockable): a pin-on-finish
  outcome shorter than what the hold could lock reports jc_pin_lockable_len.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_joint_checkpoint_mem_pressure as M  # noqa: E402
import test_joint_checkpoint_truncate as T  # noqa: E402

from sglang.srt.managers.io_struct import (  # noqa: E402
    PinOnFinishParams,
    TruncateParams,
    UnpinReqInput,
)
from sglang.srt.managers.scheduler import JcPin  # noqa: E402
from sglang.srt.mem_cache.allocation import alloc_token_slots  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT, OUTPUT, FILL = T.PROMPT, T.OUTPUT, T.FILL
SESSION = len(FILL)
SESSION_8000 = T.PREFIX_LEN + T.TAIL_LEN  # a shared-prefix session's hold


def intake_recv(rid, old_rid, pos, ttl_s):
    return SimpleNamespace(
        rid=rid,
        truncate=TruncateParams(rid=old_rid, pos=pos),
        session_params=None,
        jc_pin_on_finish=PinOnFinishParams(ttl_s=ttl_s),
    )


class TestJointCheckpointIntakeOrder(T._PinOnFinishFixture):
    def test_bad_pin_on_finish_leaves_the_truncated_request_untouched(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        succ = T.make_req("succ", FILL[:13] + [900, 901])
        err = s._jc_intake(succ, intake_recv("succ", "old", 13, ttl_s=1e9))
        self.assertIn("jc_pin_on_finish: ttl_s", err)
        self.assertIn("exceeds --jc-pin-max-ttl-s", err)
        self.assertEqual(succ.jc_pin_outcome["jc_pinned_len"], 0)
        # Not cut, no truncate hold kept for the refused successor.
        self.assert_untouched(s, old, available)
        self.assertIsNone(succ.jc_truncate_pos)
        self.assertIsNone(succ.jc_finish_hold)

    def test_valid_pin_on_finish_truncates_then_arms(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        succ = T.make_req("succ", FILL[:13] + [900, 901])
        err = s._jc_intake(succ, intake_recv("succ", "old", 13, ttl_s=30.0))
        self.assertIsNone(err)
        self.assertTrue(old.finished())
        self.assertEqual(succ.jc_truncated_at, 13)
        self.assertEqual(s.pin_table["succ"].successor_of, "old")
        self.assertIsNotNone(succ.jc_finish_hold)
        self.assertEqual(succ.jc_pin_ttl_s, 30.0)

    def test_a_refused_truncate_arms_nothing(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        succ = T.make_req("succ", [1] * 13 + [900, 901])
        err = s._jc_intake(succ, intake_recv("succ", "old", 13, ttl_s=30.0))
        self.assertIn("input_ids differ", err)
        self.assertIsNone(succ.jc_finish_hold)
        self.assertFalse(old.finished())


class TestJointCheckpointExpiredRelease(T._PinOnFinishFixture):
    def held(self, s, rid, prompt=PROMPT):
        req = self.armed(s, rid, prompt=prompt)
        self.finish(req)
        s.running_batch.filter_batch()
        self.assertIn(rid, s.pin_table)
        return req

    def test_release_after_expiry_frees_the_dead_path(self):
        s = self.make_scheduler()
        self.held(s, "a")
        self.now += s.jc_pin_ttl_s
        s._jc_expire_pins()
        self.assertNotIn("a", s.pin_table)
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertTrue(out.success)
        self.assertEqual(out.message, "request a was not pinned (expired)")
        self.assertEqual(out.released_tokens, SESSION)
        self.assertEqual(self.match_len(FILL), 0)
        self.assertEqual(self.allocator.available_size(), self.pool_size)
        # Consumed: a second unpin finds nothing.
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertEqual(out.message, "request a was not pinned")
        self.assertEqual(out.released_tokens, 0)

    def test_plain_unpin_after_expiry_frees_nothing(self):
        s = self.make_scheduler()
        self.held(s, "a")
        self.now += s.jc_pin_ttl_s
        s._jc_expire_pins()
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertEqual(out.message, "request a was not pinned (expired)")
        self.assertEqual(out.released_tokens, 0)
        self.assertEqual(self.match_len(FILL), SESSION)

    def test_expired_record_is_forgotten_after_the_max_ttl(self):
        s = self.make_scheduler()
        self.held(s, "a")
        self.now += s.jc_pin_ttl_s
        s._jc_expire_pins()
        self.now += s.jc_pin_max_ttl_s
        s._jc_expire_pins()
        self.assertEqual(s.jc_expired_pins, {})
        out = s.unpin_request(UnpinReqInput(rid="a", release=True))
        self.assertEqual(out.message, "request a was not pinned")

    def test_expired_records_are_capped_oldest_first(self):
        s = self.make_scheduler()
        s.JC_EXPIRED_PINS_MAX = 1
        self.held(s, "a", prompt=M.disjoint(0))
        s._jc_expire_pins()
        self.now += 1.0
        self.held(s, "b", prompt=M.disjoint(1))
        self.now += s.jc_pin_ttl_s
        s._jc_expire_pins()
        self.assertEqual(set(s.jc_expired_pins), {"b"})

    def test_a_new_pin_on_the_rid_drops_the_record(self):
        s = self.make_scheduler()
        self.held(s, "a")
        self.now += s.jc_pin_ttl_s
        s._jc_expire_pins()
        self.held(s, "a", prompt=M.disjoint(0))
        self.assertNotIn("a", s.jc_expired_pins)

    def test_an_expired_truncate_hold_is_not_remembered(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        _, err = self.truncate(s, "old", 13)
        self.assertIsNone(err)
        self.assertEqual(s.pin_table["succ"].successor_of, "old")
        self.now += s.JC_TRUNCATE_HOLD_TTL_S
        s._jc_expire_pins()
        self.assertEqual(s.pin_table, {})
        self.assertEqual(s.jc_expired_pins, {})


class TestJointCheckpointMemPressureIdleDrops(M._MemPressureFixture):
    def test_a_pin_held_by_a_running_request_is_kept(self):
        s = self.make_scheduler(overflow="lru")
        a = self.pinned(s, "a", 0, 1000.0)  # least recently touched: first
        self.pinned(s, "b", 1, 1001.0)
        # A running request continues a's path: dropping a's lock frees nothing.
        prompt = list(a.full_untruncated_fill_ids) + list(range(700, 720))
        r = self.running(s, "r", prompt, 64 - 2 * SESSION - 20)
        self.assertEqual(r.prefix_len, SESSION)
        self.assert_full()
        out = alloc_token_slots(self.cache, 10)
        self.assertEqual(len(out), 10)
        self.assertEqual(set(s.pin_table), {"a"})
        self.assertEqual(s.pin_table["a"].pinned_len, SESSION)
        self.assertNotIn("a", s.jc_degraded_pins)
        self.assert_mem_degraded(s, ["b"])
        # a still holds its path once r is gone.
        r.finished_reason = T.FINISH_LENGTH(length=len(r.output_ids))
        r.finished_len = len(r.output_ids)
        T.release_kv_cache(r, self.cache, checkpoint=False)
        self.evict_all()
        self.assertEqual(self.match_len(list(a.full_untruncated_fill_ids)), SESSION)

    def test_a_pin_sharing_its_path_with_a_later_drop_stays_degraded(self):
        s = self.make_scheduler(overflow="lru")
        a = self.pinned(s, "a", 0, 1000.0)
        a_pin = s.pin_table["a"]
        # A second hold on the very same path, touched after a.
        s.pin_table["a2"] = JcPin(
            lock=self.cache.lock(a_pin.lock.node),
            pinned_len=SESSION,
            pinned_at=1000.5,
            expires_at=1600.0,
            key=a_pin.key,
        )
        self.running(s, "r", M.disjoint(3), 64 - SESSION - 10)
        self.assert_full()
        alloc_token_slots(self.cache, 10)
        # Dropping a freed nothing (a2 held the path); dropping a2 freed it,
        # and the allocation evicted it: neither can be kept.
        self.assertEqual(s.pin_table, {})
        self.assert_mem_degraded(s, ["a", "a2"])
        self.assertEqual(self.match_len(list(a.full_untruncated_fill_ids)), 0)

    def test_admission_keeps_a_pin_that_made_no_room(self):
        s = self.make_scheduler(overflow="lru")
        s.truncation_align_size = None
        a = self.pinned(s, "a", 0, 1000.0)
        self.pinned(s, "b", 1, 1001.0)
        prompt = list(a.full_untruncated_fill_ids) + list(range(700, 720))
        self.running(s, "r", prompt, 64 - 2 * SESSION - 20)
        self.assert_full()
        req = T.make_req("w", M.disjoint(5)[:8])
        adder = M._LiveAdder(self.cache, need=8)
        s._jc_admit_under_memory_pressure(adder, req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(set(s.pin_table), {"a"})
        self.assert_mem_degraded(s, ["b"])


class TestJointCheckpointMemPressureIdleDropsPythonCore(
    TestJointCheckpointMemPressureIdleDrops
):
    tree_core_backend = "python"


class TestJointCheckpointFinishPinLockable(T._PinOnFinishFixture):
    def test_a_whole_hold_reports_no_lockable_length(self):
        s = self.make_scheduler()
        old = self.armed(s, "old")
        self.finish(old)
        self.assertEqual(old.jc_pin_outcome, {"jc_pinned_len": SESSION})

    def test_a_partial_hold_reports_what_it_could_lock(self):
        s = self.make_scheduler(max_ratio=0.05, overflow="lru")  # limit 12
        old = self.armed(s, "old")
        self.finish(old)
        self.assertEqual(old.jc_pin_outcome["jc_pinned_len"], 12)
        self.assertEqual(old.jc_pin_outcome["jc_pin_lockable_len"], SESSION)

    def test_a_refused_hold_reports_what_it_could_lock(self):
        s = self.make_scheduler(max_ratio=0.05, overflow="refuse")
        old = self.armed(s, "old")
        self.finish(old)
        self.assertEqual(old.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertEqual(old.jc_pin_outcome["jc_pin_lockable_len"], SESSION)

    def test_no_kv_in_the_tree_reports_no_lockable_length(self):
        s = self.make_scheduler()
        old = self.armed(s, "old")
        old.kv.cache_protected_len = 0
        s._jc_hold_on_finish(old)
        self.assertEqual(old.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertNotIn("jc_pin_lockable_len", old.jc_pin_outcome)


class TestJointCheckpointRebalanceOnFinish(T._SharedPrefixFixture):
    """--jc-pin-budget exclusive, --jc-pin-overflow next-use, limit 9000: a
    running request r1 keeps the shared prefix (4300) resident, so r0's and
    r2's pins are charged their tails (3700 each) only. When r1 finishes, the
    pins alone keep 11700: the cap is restored in valve order as r1's finish
    hold is judged, not by the next step's re-check."""

    def make_scheduler(self, *args, **kw):
        s = super().make_scheduler(*args, max_ratio=9000 / 32768, **kw)
        s.jc_pin_budget = "exclusive"
        return s

    def drop_finished(self, s):
        s.running_batch.reqs = [r for r in s.running_batch.reqs if not r.finished()]

    def test_the_cap_holds_when_a_shared_prefix_is_charged_again(self):
        s = self.make_scheduler(overflow="next-use")
        limit = int(s.jc_pin_max_ratio * s.max_total_num_tokens)
        self.now = 1000.0
        r0 = self.armed(s, "r0", prompt=T.session_prompt(0), outputs=[7], ttl_s=100.0)
        self.finish(r0)  # expires 1100
        self.drop_finished(s)
        # r1 runs on the shared prefix; its finish hold is used last (expiry 4000).
        r1 = self.armed(s, "r1", prompt=T.session_prompt(1), outputs=[7], ttl_s=3000.0)
        self.assertEqual(r1.prefix_len, T.PREFIX_LEN)
        self.now = 1001.0
        r2 = self.armed(s, "r2", prompt=T.session_prompt(2), outputs=[7], ttl_s=100.0)
        self.finish(r2)  # expires 1101
        self.drop_finished(s)
        s._jc_enforce_pin_cap()
        self.assertEqual(s._jc_pinned_tokens_total(), 2 * T.TAIL_LEN)
        self.assertEqual(s.jc_pins_degraded_total, 0)
        self.now = 1002.0
        self.finish(r1)
        self.drop_finished(s)
        # In this step: r2 (expected to be used latest of r0, r2) gave way,
        # then r1's hold took what fits beside r0.
        self.assertLessEqual(s._jc_pinned_tokens_total(), limit)
        self.assertEqual(set(s.pin_table), {"r0", "r1"})
        self.assertIn("r2", s.jc_degraded_pins)
        self.assertEqual(s.jc_pins_degraded_total, 1)
        # r0 keeps 8000; r1 adds its tail up to the 1000 left
        self.assertEqual(r1.jc_pin_outcome["jc_pinned_len"], T.PREFIX_LEN + limit - SESSION_8000)
        self.assertEqual(r1.jc_pin_outcome["jc_pin_lockable_len"], SESSION_8000)
        # The next step's re-check finds nothing to do.
        s._jc_enforce_pin_cap()
        self.assertEqual(s.jc_pins_degraded_total, 1)
        self.assertEqual(set(s.pin_table), {"r0", "r1"})

    def test_unchanged_in_flight_set_degrades_nothing_more(self):
        s = self.make_scheduler(overflow="next-use")
        self.now = 1000.0
        r0 = self.armed(s, "r0", prompt=T.session_prompt(0), outputs=[7], ttl_s=100.0)
        self.finish(r0)
        self.drop_finished(s)
        self.now = 1001.0
        r2 = self.armed(s, "r2", prompt=T.session_prompt(2), outputs=[7], ttl_s=100.0)
        self.finish(r2)  # 8000 + 3700 > 9000: a partial hold, nothing degraded
        self.assertEqual(s.jc_pins_degraded_total, 0)
        self.assertEqual(set(s.pin_table), {"r0", "r2"})
        self.assertLessEqual(
            s._jc_pinned_tokens_total(), int(s.jc_pin_max_ratio * s.max_total_num_tokens)
        )


class TestJointCheckpointNoHoldBesideAnOverCapSet(T._SharedPrefixFixture):
    """A hold is not taken beside holds that exceed the cap by themselves
    (here truncate holds, which no valve degrades), even one that would add
    no token to them."""

    def test_refused_instead_of_a_free_riding_partial_hold(self):
        s = self.make_scheduler(max_ratio=9000 / 32768, overflow="lru")
        r0 = self.armed(s, "r0", prompt=T.session_prompt(0), outputs=[7])
        self.finish(r0)
        r2 = self.start("r2", T.session_prompt(2), [7])
        self.finish(r2)
        self.drop = [r for r in s.running_batch.reqs if not r.finished()]
        s.running_batch.reqs = self.drop
        for rid, req in (("h0", r0), ("h2", r2)):
            s.pin_table[rid] = JcPin(
                lock=self.cache.lock(req.last_node),
                pinned_len=SESSION_8000,
                pinned_at=900.0,
                expires_at=1030.0,
                successor_of="x",
            )
        self.assertGreater(
            s._jc_pinned_tokens_total(exclude={"r0"}),
            int(s.jc_pin_max_ratio * s.max_total_num_tokens),
        )
        r1 = self.armed(s, "r1", prompt=T.session_prompt(1), outputs=[7])
        self.finish(r1)
        self.assertEqual(r1.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertIn("the other holds alone keep", r1.jc_pin_outcome["jc_pin_message"])
        self.assertEqual(set(s.pin_table), {"r0", "h0", "h2"})
        self.assertEqual(s.jc_pins_degraded_total, 0)


if __name__ == "__main__":
    unittest.main()
