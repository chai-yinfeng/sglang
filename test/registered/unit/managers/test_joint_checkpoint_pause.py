"""Joint checkpoint P5 (pause / resume / watermark budget) on a CPU-only
Scheduler built with Scheduler.__new__, real Req and ScheduleBatch objects.

- Scheduler._jc_pause_resume moves requests between the running batch and
  Scheduler.paused_reqs without touching their KV, rows or tree state.
- Req.jc_over_budget / jc_wants_pause decide the lookahead budget.
- A pause longer than --jc-pause-timeout-s turns into an abort finish.
- abort_request on a paused request marks it to finish and resumes it, so it
  takes the running request's one-decode-step abort path.
The resume batch builder (_build_hisparse_decode_batch, needs a future map)
is replaced by a CPU stand-in that fills the same tensors.
"""

import unittest
from array import array
from collections import deque
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    PauseReqInput,
    ResumeReqInput,
    WatermarkReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT_LEN = 3


def make_req(rid, output_len, lookahead=None, row=0) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=SamplingParams(lookahead=lookahead, max_new_tokens=1000),
    )
    req.output_ids.extend(range(100, 100 + output_len))
    req.kv.req_pool_idx = row
    return req


class TestJointCheckpointPause(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    # ------------------------------------------------------------ fixtures

    def make_batch(self, scheduler, reqs) -> ScheduleBatch:
        batch = ScheduleBatch(reqs=list(reqs), batch_is_full=False)
        batch.device = "cpu"
        batch.req_to_token_pool = scheduler.req_to_token_pool
        batch.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        batch.tree_cache = scheduler.tree_cache
        batch.model_config = MagicMock(is_encoder_decoder=False)
        batch.sampling_info = MagicMock()
        batch.spec_info = None
        batch.multimodal_inputs = None
        rows = [r.kv.req_pool_idx for r in reqs]
        lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.seq_lens = torch.tensor(lens, dtype=torch.int64)
        batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(lens, dtype=torch.int32)
        batch.input_ids = None
        return batch

    def make_scheduler(self, reqs, enable=True, timeout_s=300.0) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.jc_pause_timeout_s = timeout_s
        s.paused_reqs = {}
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.tree_cache = MagicMock()
        s.req_to_token_pool = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.running_batch = self.make_batch(s, reqs)
        s.last_batch = None
        s.chunked_req = None
        s.result_queue = deque()
        s.enable_continuous_input_polling = False
        # abort_request collaborators that must stay out of the way.
        s.mm_receiver = None
        s.waiting_queue = []
        s.dllm_config = None
        s.grammar_manager = MagicMock()
        s.disaggregation_mode = DisaggregationMode.NULL
        s._pending_chunked_abort_req = None
        # The real builder stashes into the GPU future map; on CPU fill the
        # same per-request tensors.
        s._build_hisparse_decode_batch = MagicMock(
            side_effect=lambda reqs: self.make_batch(s, reqs)
        )
        return s

    def step(self, s):
        """One scheduling step's worth of pause/resume bookkeeping."""
        s.running_batch = s._jc_pause_resume(s.running_batch)
        return s.running_batch

    def assert_kv_untouched(self, s, req, row):
        self.assertEqual(req.kv.req_pool_idx, row)
        self.assertFalse(req.finished())
        s.req_to_token_pool.free.assert_not_called()
        s.tree_cache.free_kv_row.assert_not_called()
        s.tree_cache.finish.assert_not_called()
        s.token_to_kv_pool_allocator.free.assert_not_called()

    # ------------------------------------------------------------ budget predicate

    def test_lookahead_boundary(self):
        req = make_req("a", output_len=6, lookahead=4)
        req.jc_watermark = 2  # 6 - 2 == 4: at the budget, not over it
        self.assertFalse(req.jc_over_budget())
        self.assertFalse(req.jc_wants_pause())
        req.jc_watermark = 1  # 6 - 1 == 5 > 4
        self.assertTrue(req.jc_over_budget())
        self.assertTrue(req.jc_wants_pause())
        # No lookahead: never over budget.
        self.assertFalse(make_req("b", output_len=50).jc_over_budget())
        # lookahead 0 pauses as soon as anything beyond the watermark exists.
        zero = make_req("c", output_len=1, lookahead=0)
        self.assertTrue(zero.jc_wants_pause())
        zero.jc_watermark = 1
        self.assertFalse(zero.jc_wants_pause())

    def test_wants_pause_needs_an_output_token_and_no_finish(self):
        fresh = make_req("a", output_len=0, lookahead=0)
        fresh.jc_pause_reason = "manual"
        self.assertFalse(fresh.jc_wants_pause(), "nothing to resume from yet")
        finishing = make_req("b", output_len=9, lookahead=1)
        self.assertTrue(finishing.jc_wants_pause())
        finishing.to_finish = FINISH_ABORT()
        self.assertFalse(finishing.jc_wants_pause(), "a finishing request must run")

    # ------------------------------------------------------------ pause / resume

    def test_budget_pause_leaves_running_batch_and_keeps_kv(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        b = make_req("b", output_len=6, row=1)
        s = self.make_scheduler([a, b])
        batch = self.step(s)
        self.assertEqual([r.rid for r in batch.reqs], ["b"])
        self.assertEqual(batch.req_pool_indices.tolist(), [1])
        self.assertEqual(batch.seq_lens.tolist(), [PROMPT_LEN + 6 - 1])
        self.assertFalse(batch.batch_is_full)
        self.assertIs(s.paused_reqs["a"], a)
        self.assertEqual(a.jc_pause_reason, "budget")
        self.assertIsNotNone(a.jc_paused_at)
        self.assert_kv_untouched(s, a, row=0)
        # Idempotent while nothing changes.
        self.step(s)
        self.assertEqual([r.rid for r in s.running_batch.reqs], ["b"])
        self.assertEqual(list(s.paused_reqs), ["a"])

    def test_watermark_within_budget_resumes_into_running_batch(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        b = make_req("b", output_len=6, row=1)
        s = self.make_scheduler([a, b])
        self.step(s)
        out = s.watermark_request(WatermarkReqInput(rid="a", watermark=2))
        self.assertTrue(out.success)
        self.assertEqual((out.output_len, out.paused), (6, True))
        batch = self.step(s)
        self.assertEqual(sorted(r.rid for r in batch.reqs), ["a", "b"])
        self.assertEqual(sorted(batch.req_pool_indices.tolist()), [0, 1])
        self.assertEqual(s.paused_reqs, {})
        self.assertIsNone(a.jc_pause_reason)
        self.assertIsNone(a.jc_paused_at)
        s._build_hisparse_decode_batch.assert_called_once_with([a])
        self.assert_kv_untouched(s, a, row=0)

    def test_resume_into_empty_running_batch(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        s = self.make_scheduler([a])
        self.assertTrue(self.step(s).is_empty())
        s.watermark_request(WatermarkReqInput(rid="a", watermark=6))
        batch = self.step(s)
        self.assertEqual([r.rid for r in batch.reqs], ["a"])
        self.assertEqual(batch.req_pool_indices.tolist(), [0])

    def test_watermark_is_monotonic_and_clamped(self):
        a = make_req("a", output_len=6, lookahead=4)
        s = self.make_scheduler([a])
        s.watermark_request(WatermarkReqInput(rid="a", watermark=9))
        self.assertEqual(a.jc_watermark, 6)
        s.watermark_request(WatermarkReqInput(rid="a", watermark=3))
        self.assertEqual(a.jc_watermark, 6)

    def test_manual_pause_sticks_until_resume(self):
        a = make_req("a", output_len=6, row=0)  # no lookahead
        s = self.make_scheduler([a])
        out = s.pause_request(PauseReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertFalse(out.paused, "takes effect at the next scheduling step")
        self.assertTrue(self.step(s).is_empty())
        self.assertEqual(a.jc_pause_reason, "manual")
        # A watermark does not resume a manual pause.
        s.watermark_request(WatermarkReqInput(rid="a", watermark=6))
        self.assertTrue(self.step(s).is_empty())
        out = s.resume_request(ResumeReqInput(rid="a"))
        self.assertTrue(out.success and out.paused)
        batch = self.step(s)
        self.assertEqual([r.rid for r in batch.reqs], ["a"])
        self.assert_kv_untouched(s, a, row=0)

    def test_resume_while_over_budget_pauses_again(self):
        a = make_req("a", output_len=6, lookahead=4)
        s = self.make_scheduler([a])
        self.step(s)
        out = s.resume_request(ResumeReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertIn("over the lookahead budget", out.message)
        self.assertTrue(self.step(s).is_empty())
        self.assertEqual(a.jc_pause_reason, "budget")

    # ------------------------------------------------------------ timeout / abort

    def test_pause_timeout_aborts_and_resumes_to_finish(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        s = self.make_scheduler([a], timeout_s=10.0)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=1000.0):
            self.step(s)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=1009.0):
            self.assertTrue(self.step(s).is_empty(), "within the timeout")
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=1010.5):
            batch = self.step(s)
        self.assertEqual([r.rid for r in batch.reqs], ["a"])
        self.assertIsInstance(a.to_finish, FINISH_ABORT)
        self.assertIn("pause timeout", a.to_finish.message)
        self.assertEqual(s.paused_reqs, {})
        # Not finished here: the decode step that follows finishes it.
        self.assertFalse(a.finished())

    def test_abort_paused_request_marks_finish_and_resumes(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        s = self.make_scheduler([a])
        self.step(s)
        s.abort_request(AbortReq(rid="a", discard=True))
        self.assertIsInstance(a.to_finish, FINISH_ABORT)
        self.assertTrue(a.skip_radix_cache_insert)
        batch = self.step(s)
        self.assertEqual([r.rid for r in batch.reqs], ["a"])
        self.assertEqual(s.paused_reqs, {})

    def test_paused_request_finished_in_flight_is_dropped(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        s = self.make_scheduler([a])
        self.step(s)
        # The overlapped result that was in flight at pause time finished it.
        a.finished_reason = FINISH_ABORT()
        self.assertTrue(self.step(s).is_empty())
        self.assertEqual(s.paused_reqs, {})
        s._build_hisparse_decode_batch.assert_not_called()

    def test_abort_all_covers_paused_requests(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        s = self.make_scheduler([a])
        self.step(s)
        s.abort_request(AbortReq(abort_all=True))
        self.assertIsInstance(a.to_finish, FINISH_ABORT)
        self.assertFalse(a.skip_radix_cache_insert)

    # ------------------------------------------------------------ guards

    def test_control_messages_rejected_when_flag_off(self):
        a = make_req("a", output_len=6, lookahead=4)
        s = self.make_scheduler([a], enable=False)
        for out in (
            s.pause_request(PauseReqInput(rid="a")),
            s.resume_request(ResumeReqInput(rid="a")),
            s.watermark_request(WatermarkReqInput(rid="a", watermark=1)),
        ):
            self.assertFalse(out.success)
            self.assertIn("--enable-joint-checkpoint", out.message)
        self.assertIsNone(a.jc_pause_reason)

    def test_control_messages_rejected_with_speculative_decoding(self):
        a = make_req("a", output_len=6, lookahead=4)
        s = self.make_scheduler([a])
        s.spec_algorithm = SpeculativeAlgorithm.EAGLE
        out = s.pause_request(PauseReqInput(rid="a"))
        self.assertFalse(out.success)
        self.assertIn("speculative", out.message)
        # And the budget never pauses either.
        self.assertEqual([r.rid for r in self.step(s).reqs], ["a"])

    def test_unknown_or_finishing_request(self):
        a = make_req("a", output_len=6)
        s = self.make_scheduler([a])
        self.assertIn("not running", s.pause_request(PauseReqInput(rid="zz")).message)
        a.to_finish = FINISH_ABORT()
        self.assertIn("finishing", s.pause_request(PauseReqInput(rid="a")).message)

    def test_sampling_params_reject_negative_lookahead(self):
        with self.assertRaises(ValueError):
            SamplingParams(lookahead=-1).verify(vocab_size=100)
        SamplingParams(lookahead=0).verify(vocab_size=100)


if __name__ == "__main__":
    unittest.main()
