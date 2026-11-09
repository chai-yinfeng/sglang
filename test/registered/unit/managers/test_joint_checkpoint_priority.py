"""Joint checkpoint P4 (runtime set_priority) on a CPU-only Scheduler built
with Scheduler.__new__ and real Req objects.

- Scheduler.set_priority rewrites req.priority for a waiting, running or
  paused request and reports old/new; it is refused when the joint-checkpoint
  flag or priority scheduling is off, or the rid is unknown.
- The two consumers read req.priority directly, so the change is visible to
  SchedulePolicy._sort_by_priority_and_fcfs (waiting queue) and to
  PrefillAdder.preempt_to_schedule (running requests) on their next call.
- A raise is forwarded to tree_cache.raise_path_priority for a request that
  holds KV; a lower value is not.
"""

import unittest
from array import array
from unittest.mock import MagicMock, PropertyMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import SetPriorityReqInput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Default: larger value = higher priority (schedule_low_priority_values_first off).
PRIORITY_SIGN = -1


def make_req(rid, priority, entry_time=0.0, row=None) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=SamplingParams(max_new_tokens=16),
        priority=priority,
    )
    req.time_stats.wait_queue_entry_time = entry_time
    req.kv.req_pool_idx = row
    return req


class TestJointCheckpointPriority(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    # ------------------------------------------------------------ fixtures

    def make_scheduler(
        self, waiting=(), running=(), paused=(), enable=True, priority_sched=True
    ) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.enable_priority_scheduling = priority_sched
        s.paused_reqs = {r.rid: r for r in paused}
        s.waiting_queue = list(waiting)
        s.running_batch = ScheduleBatch(reqs=list(running), batch_is_full=False)
        s.last_batch = None
        s.chunked_req = None
        s.tree_cache = MagicMock()
        s.tree_cache.raise_path_priority.return_value = True
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_continuous_input_polling = False
        return s

    def set_priority(self, s, rid, priority):
        return s.set_priority(SetPriorityReqInput(rid=rid, priority=priority))

    # ------------------------------------------------------------ guards

    def test_refused_without_joint_checkpoint_flag(self):
        req = make_req("a", 0)
        s = self.make_scheduler(waiting=[req], enable=False)
        out = self.set_priority(s, "a", 5)
        self.assertFalse(out.success)
        self.assertIn("--enable-joint-checkpoint", out.message)
        self.assertEqual(req.priority, 0)

    def test_refused_without_priority_scheduling(self):
        req = make_req("a", None)
        s = self.make_scheduler(waiting=[req], priority_sched=False)
        out = self.set_priority(s, "a", 5)
        self.assertFalse(out.success)
        self.assertIn("--enable-priority-scheduling", out.message)
        self.assertIsNone(req.priority)

    def test_refused_for_unknown_rid(self):
        s = self.make_scheduler(waiting=[make_req("a", 0)], running=[make_req("b", 0, row=0)])
        out = self.set_priority(s, "zzz", 5)
        self.assertFalse(out.success)
        self.assertIn("zzz", out.message)

    # ------------------------------------------------------------ waiting queue

    def test_waiting_request_reorders_on_next_sort(self):
        lo = make_req("lo", 0, entry_time=1.0)
        hi = make_req("hi", 100, entry_time=2.0)
        s = self.make_scheduler(waiting=[lo, hi])

        SchedulePolicy._sort_by_priority_and_fcfs(s.waiting_queue, PRIORITY_SIGN)
        self.assertEqual([r.rid for r in s.waiting_queue], ["hi", "lo"])

        out = self.set_priority(s, "lo", 200)
        self.assertTrue(out.success)
        self.assertEqual((out.old_priority, out.new_priority), (0, 200))
        self.assertEqual(lo.priority, 200)
        # Not in the tree: nothing to raise.
        s.tree_cache.raise_path_priority.assert_not_called()

        SchedulePolicy._sort_by_priority_and_fcfs(s.waiting_queue, PRIORITY_SIGN)
        self.assertEqual([r.rid for r in s.waiting_queue], ["lo", "hi"])

    # ------------------------------------------------------------ running / preemption

    def make_adder(self, running_batch, threshold=0) -> PrefillAdder:
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.running_batch = running_batch
        adder.preempt_list = []
        adder.priority_scheduling_preemption_threshold = threshold
        adder.memory_budget = MagicMock(total_offset=0)
        adder._request_slot_reserve_for_req = MagicMock(return_value=0)
        adder._get_running_request_total_token_offset = MagicMock(return_value=100)
        return adder

    def preemptible(self, s, new_req) -> bool:
        """Does preempt_to_schedule(new_req) take a running request out?"""
        batch = MagicMock()
        batch.reqs = list(s.running_batch.reqs)
        adder = self.make_adder(batch)
        with patch.object(PrefillAdder, "rem_total_tokens", new_callable=PropertyMock) as rem:
            rem.return_value = 0  # the new request needs tokens from a running one
            return adder.preempt_to_schedule(new_req)

    def test_running_request_preemptibility_follows_new_priority(self):
        running = make_req("run", 100, entry_time=1.0, row=0)
        running.prefix_len = 0
        s = self.make_scheduler(running=[running])
        newcomer = make_req("new", 100, entry_time=5.0)
        newcomer.prefix_len = 0

        # Equal priority: the newcomer cannot preempt.
        self.assertFalse(self.preemptible(s, newcomer))

        out = self.set_priority(s, "run", 0)
        self.assertTrue(out.success)
        self.assertEqual((out.old_priority, out.new_priority), (100, 0))
        self.assertIn("lowered", out.message)
        # A lower value is not written to shared tree nodes.
        s.tree_cache.raise_path_priority.assert_not_called()

        # Now the running request is below the newcomer and can be preempted.
        self.assertTrue(self.preemptible(s, newcomer))

        # Raising it back above the threshold protects it again.
        out = self.set_priority(s, "run", 100)
        self.assertTrue(out.success)
        self.assertFalse(self.preemptible(s, newcomer))

    def test_raise_on_running_request_reaches_tree_path(self):
        running = make_req("run", 0, row=0)
        running.last_node = 42
        s = self.make_scheduler(running=[running])
        out = self.set_priority(s, "run", 7)
        self.assertTrue(out.success)
        self.assertIn("raised", out.message)
        s.tree_cache.raise_path_priority.assert_called_once_with(42, 7)

    def test_raise_reports_when_tree_core_cannot_update(self):
        running = make_req("run", 0, row=0)
        running.last_node = 42
        s = self.make_scheduler(running=[running])
        s.tree_cache.raise_path_priority.return_value = False
        out = self.set_priority(s, "run", 7)
        self.assertTrue(out.success)
        self.assertEqual(running.priority, 7)
        self.assertIn("not updated in place", out.message)

    def test_paused_request_is_found(self):
        paused = make_req("p", 0, row=1)
        s = self.make_scheduler(paused=[paused])
        out = self.set_priority(s, "p", 3)
        self.assertTrue(out.success)
        self.assertEqual(paused.priority, 3)


if __name__ == "__main__":
    unittest.main()
