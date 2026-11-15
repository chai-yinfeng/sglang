"""Joint checkpoint P5 (pause / resume) under NGRAM speculative decoding, on a
CPU-only Scheduler built with Scheduler.__new__, real Req / ScheduleBatch
objects, a real NGRAMWorker (object.__new__) with a real NgramCorpus, and a
real FutureMap (object.__new__, CPU buffers only).

- A pause erases the request's per-rid corpus match state; the shared trie
  keeps every n-gram.
- The resume batch carries the spec input an NGRAM request carries from its
  extend step into its first decode (built here by the worker's own extend
  branch), with nothing accepted pending, and the next draft prep drafts from
  the request's own tail.
- Manual pause / resume works under NGRAM; other algorithms stay rejected
  with their name in the message; a pause timeout still turns into an abort.
"""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    PauseReqInput,
    ResumeReqInput,
    WatermarkReqInput,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PROMPT = [1, 2, 3]
DRAFT_TOKENS = 4
TRIE_DEPTH = 8
POOL_ROWS = 8
VOCAB = 1000
# A sequence whose continuation of request "a"'s tail (outputs 100..105) is
# 106, 107, 108: what a draft for "a" should propose.
CORPUS = [1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 108]


def make_req(rid, output_len, lookahead=None, row=0, **kwargs) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", PROMPT),
        sampling_params=SamplingParams(lookahead=lookahead, max_new_tokens=1000),
        **kwargs,
    )
    req.output_ids.extend(range(100, 100 + output_len))
    req.kv.req_pool_idx = row
    return req


def make_corpus() -> NgramCorpus:
    corpus = NgramCorpus(
        max_trie_depth=TRIE_DEPTH,
        min_bfs_breadth=1,
        max_bfs_breadth=1,
        draft_token_num=DRAFT_TOKENS,
        capacity=1000,
    )
    corpus.batch_put([CORPUS])
    corpus.synchronize()
    return corpus


def make_worker(overlap) -> NGRAMWorker:
    worker = object.__new__(NGRAMWorker)
    worker.enable_overlap = overlap
    worker.device = "cpu"
    worker.max_trie_depth = TRIE_DEPTH
    worker.draft_token_num = DRAFT_TOKENS
    worker.speculative_num_draft_tokens = DRAFT_TOKENS
    worker._prev_decode_rids = set()
    worker.ngram_corpus = make_corpus()
    worker._target_worker = MagicMock()
    return worker


def make_future_map() -> FutureMap:
    fm = object.__new__(FutureMap)
    fm.spec_algo = SpeculativeAlgorithm.NGRAM
    fm.device = "cpu"
    fm.req_pool_size = POOL_ROWS
    fm.accept_tokens_buf = None
    fm.accept_lens_buf = None
    return fm


def probe_trie(corpus, tokens):
    """Drafts for ``tokens`` from a throwaway match state: reads the trie only."""
    drafts, mask = corpus.batch_get(["probe"], [tokens], [len(tokens)])
    corpus.erase_match_state(["probe"])
    return drafts.tolist(), mask.tolist()


class TestJointCheckpointPauseNgram(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(
            ServerArgs(model_path="dummy", speculative_num_draft_tokens=DRAFT_TOKENS),
            role="test",
        )
        self.addCleanup(reset_context)

    # ------------------------------------------------------------ fixtures

    def make_batch(self, s, reqs, forward_mode=ForwardMode.DECODE) -> ScheduleBatch:
        batch = ScheduleBatch(reqs=list(reqs), batch_is_full=False)
        batch.device = "cpu"
        batch.req_to_token_pool = s.req_to_token_pool
        batch.token_to_kv_pool_allocator = s.token_to_kv_pool_allocator
        batch.tree_cache = s.tree_cache
        batch.model_config = s.model_config
        batch.spec_algorithm = s.spec_algorithm
        batch.enable_overlap = s.enable_overlap
        batch.forward_mode = forward_mode
        batch.sampling_info = MagicMock()
        batch.spec_info = None
        batch.multimodal_inputs = [None] * len(reqs)
        batch.return_logprob = False
        rows = [r.kv.req_pool_idx for r in reqs]
        lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.seq_lens = torch.tensor(lens, dtype=torch.int64)
        batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(lens, dtype=torch.int32)
        batch.input_ids = None
        return batch

    def run_extend_step(self, s, batch) -> NgramVerifyInput:
        """The upstream prefill -> first-decode path: the worker's extend
        branch builds next_draft_input, the scheduler installs it (run_batch)
        and, under overlap, relays it (_relay_forward_payload)."""
        bs = len(batch.reqs)
        s.draft_worker._target_worker.forward_batch_generation.return_value = (
            SimpleNamespace(
                logits_output=None,
                next_token_ids=torch.arange(500, 500 + bs, dtype=torch.int32),
                can_run_cuda_graph=False,
            )
        )
        result = s.draft_worker.forward_batch_generation(batch)
        draft_input = result.next_draft_input
        if s.enable_overlap:
            s.future_map.stash(
                batch.req_pool_indices, RelayPayload.from_ngram(draft_input)
            )
            draft_input.future_indices = batch.req_pool_indices
        return draft_input

    def make_scheduler(self, reqs, overlap=True, timeout_s=300.0) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = True
        s.jc_pause_timeout_s = timeout_s
        s.paused_reqs = {}
        s.spec_algorithm = SpeculativeAlgorithm.NGRAM
        s.enable_overlap = overlap
        s.device = "cpu"
        s.model_config = MagicMock(is_encoder_decoder=False, vocab_size=VOCAB)
        s.draft_worker = make_worker(overlap)
        s.future_map = make_future_map()
        s.tree_cache = MagicMock()
        s.req_to_token_pool = MagicMock(device="cpu")
        s.token_to_kv_pool_allocator = MagicMock()
        s.last_batch = None
        s.chunked_req = None
        s.result_queue = deque()
        s.enable_continuous_input_polling = False
        s.waiting_queue = []
        s.dllm_config = None
        s.disaggregation_mode = DisaggregationMode.NULL
        batch = self.make_batch(s, reqs, forward_mode=ForwardMode.EXTEND)
        batch.spec_info = self.run_extend_step(s, batch)
        batch.forward_mode = ForwardMode.DECODE
        s.running_batch = batch
        return s

    def step(self, s):
        s.running_batch = s._jc_pause_resume(s.running_batch)
        return s.running_batch

    def seed_match_states(self, s, reqs):
        """Give each request a corpus match state, as a verify step would."""
        tokens = [list(r.origin_input_ids) + list(r.output_ids) for r in reqs]
        s.draft_worker.ngram_corpus.batch_get(
            [r.rid for r in reqs], tokens, [len(t) for t in tokens]
        )

    def match_rids(self, s):
        return set(s.draft_worker.ngram_corpus._req_id_to_state_id)

    def spy_resume_builder(self, s):
        spy = MagicMock(wraps=s._jc_build_ngram_resume_batch)
        s._jc_build_ngram_resume_batch = spy
        return spy

    # ------------------------------------------------------------ pause

    def test_budget_pause_erases_match_state_and_keeps_trie(self):
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                a = make_req("a", output_len=6, lookahead=4, row=0)
                b = make_req("b", output_len=6, row=1)
                s = self.make_scheduler([a, b], overlap=overlap)
                self.seed_match_states(s, [a, b])
                self.assertEqual(self.match_rids(s), {"a", "b"})
                trie_before = probe_trie(s.draft_worker.ngram_corpus, PROMPT)

                batch = self.step(s)
                self.assertEqual([r.rid for r in batch.reqs], ["b"])
                self.assertIs(s.paused_reqs["a"], a)
                self.assertEqual(self.match_rids(s), {"b"})
                self.assertEqual(
                    probe_trie(s.draft_worker.ngram_corpus, PROMPT), trie_before
                )
                # spec_info follows the filter.
                spec = batch.spec_info
                if overlap:
                    self.assertEqual(spec.future_indices.tolist(), [1])
                else:
                    self.assertIsNone(spec.future_indices)
                    self.assertEqual(spec.accept_lens.shape, (1,))
                    self.assertEqual(spec.accept_tokens.numel(), DRAFT_TOKENS)
                    self.assertEqual(spec.new_seq_lens.shape, (1,))
                # The worker's own departure bookkeeping erasing it again later
                # is harmless.
                s.draft_worker.ngram_corpus.erase_match_state(["a"])
                self.assertEqual(self.match_rids(s), {"b"})

    # ------------------------------------------------------------ resume

    def test_resume_batch_matches_extend_to_decode_input(self):
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                a = make_req("a", output_len=6, lookahead=4, row=0)
                b = make_req("b", output_len=6, row=1)
                s = self.make_scheduler([a, b], overlap=overlap)
                reference = s.running_batch.spec_info  # from the extend branch
                ref_tokens = reference.accept_tokens
                ref_lens = reference.accept_lens
                ref_seq_lens = reference.new_seq_lens
                self.step(s)
                spy = self.spy_resume_builder(s)
                s.watermark_request(WatermarkReqInput(rid="a", watermark=2))
                trie_before = probe_trie(s.draft_worker.ngram_corpus, PROMPT)
                batch = self.step(s)

                spy.assert_called_once_with([a])
                self.assertEqual([r.rid for r in batch.reqs], ["b", "a"])
                self.assertEqual(s.paused_reqs, {})
                self.assertEqual(batch.req_pool_indices.tolist(), [1, 0])
                self.assertEqual(
                    batch.seq_lens.tolist(), [len(PROMPT) + 6 - 1] * 2
                )
                self.assertEqual(len(batch.multimodal_inputs), 2)
                spec = batch.spec_info
                self.assertIsInstance(spec, NgramVerifyInput)
                self.assertEqual(spec.draft_token_num, DRAFT_TOKENS)
                if overlap:
                    self.assertEqual(spec.future_indices.tolist(), [1, 0])
                    rows = torch.tensor([0])
                    self.assertEqual(s.future_map.accept_lens_buf[rows].tolist(), [0])
                    self.assertEqual(
                        s.future_map.accept_tokens_buf.dtype, ref_tokens.dtype
                    )
                    self.assertEqual(
                        s.future_map.accept_lens_buf.dtype, ref_lens.dtype
                    )
                else:
                    self.assertIsNone(spec.future_indices)
                    self.assertEqual(spec.accept_lens.tolist()[1], 0)
                    self.assertEqual(spec.accept_lens.dtype, ref_lens.dtype)
                    self.assertEqual(spec.accept_tokens.dtype, ref_tokens.dtype)
                    self.assertEqual(spec.accept_tokens.numel(), 2 * DRAFT_TOKENS)
                    self.assertEqual(spec.new_seq_lens.dtype, ref_seq_lens.dtype)
                    self.assertEqual(spec.new_seq_lens.tolist(), [8, 8])
                self.assertEqual(
                    probe_trie(s.draft_worker.ngram_corpus, PROMPT), trie_before
                )

    def test_resume_builder_matches_extend_branch_field_by_field(self):
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                a = make_req("a", output_len=6, row=0)
                s = self.make_scheduler([make_req("x", output_len=1, row=3)], overlap)
                reference = s.running_batch.spec_info
                built = s._jc_build_ngram_resume_batch([a])
                spec = built.spec_info
                self.assertIs(type(spec), type(reference))
                for name in ("accept_tokens", "accept_lens", "new_seq_lens"):
                    got, ref = getattr(spec, name), getattr(reference, name)
                    self.assertEqual((got.dtype, got.shape), (ref.dtype, ref.shape))
                self.assertEqual(spec.draft_token_num, reference.draft_token_num)
                self.assertEqual(
                    spec.future_indices is None, reference.future_indices is None
                )
                self.assertEqual(spec.accept_lens.tolist(), [0])
                self.assertEqual(built.seq_lens.tolist(), [len(PROMPT) + 6 - 1])
                self.assertEqual(built.seq_lens_cpu.tolist(), [len(PROMPT) + 6 - 1])
                self.assertEqual(built.req_pool_indices.tolist(), [0])
                self.assertIsNone(built.input_ids)
                self.assertIs(built.spec_algorithm, SpeculativeAlgorithm.NGRAM)
                # Merges with the extend-born running batch (per-request lists
                # are shaped by _jc_pause_resume after the builder).
                built.multimodal_inputs = [None]
                s.running_batch.merge_batch(built)
                self.assertEqual(
                    [r.rid for r in s.running_batch.reqs], ["x", "a"]
                )

    def test_next_draft_after_resume_starts_from_request_tail(self):
        for overlap in (True, False):
            with self.subTest(overlap=overlap):
                a = make_req("a", output_len=6, lookahead=4, row=0)
                b = make_req("b", output_len=6, row=1)
                s = self.make_scheduler([a, b], overlap=overlap)
                self.seed_match_states(s, [a, b])
                self.step(s)
                s.watermark_request(WatermarkReqInput(rid="a", watermark=2))
                batch = self.step(s)
                self.assertEqual([r.rid for r in batch.reqs], ["b", "a"])
                if overlap:
                    # resolve_forward_inputs' NGRAM relay gather.
                    s.future_map._resolve_spec_extras(batch)
                drafts, mask = s.draft_worker._prepare_draft_tokens(batch)
                self.assertEqual(drafts.shape, (2 * DRAFT_TOKENS,))
                self.assertEqual(mask.shape, (2 * DRAFT_TOKENS * DRAFT_TOKENS,))
                # "a" drafts from its own tail, nothing spliced: what a fresh
                # corpus with the same trie proposes for prompt + output_ids.
                tokens = PROMPT + list(a.output_ids)
                ref_drafts, ref_mask = make_corpus().batch_get(
                    ["a"], [tokens], [len(tokens)]
                )
                np.testing.assert_array_equal(drafts[DRAFT_TOKENS:], ref_drafts)
                np.testing.assert_array_equal(
                    mask[DRAFT_TOKENS * DRAFT_TOKENS :], ref_mask
                )
                self.assertEqual(int(drafts[DRAFT_TOKENS]), a.output_ids[-1])
                self.assertEqual(drafts[DRAFT_TOKENS:].tolist()[1:], [106, 107, 108])
                self.assertIn("a", self.match_rids(s))

    # ------------------------------------------------------------ control

    def test_manual_pause_and_resume_under_ngram(self):
        a = make_req("a", output_len=6, row=0)
        s = self.make_scheduler([a])
        self.seed_match_states(s, [a])
        out = s.pause_request(PauseReqInput(rid="a"))
        self.assertTrue(out.success, out.message)
        self.assertTrue(self.step(s).is_empty())
        self.assertEqual(a.jc_pause_reason, "manual")
        self.assertNotIn("a", self.match_rids(s))
        out = s.resume_request(ResumeReqInput(rid="a"))
        self.assertTrue(out.success and out.paused, out.message)
        spy = self.spy_resume_builder(s)
        batch = self.step(s)
        spy.assert_called_once_with([a])
        self.assertEqual([r.rid for r in batch.reqs], ["a"])
        self.assertIsInstance(batch.spec_info, NgramVerifyInput)
        self.assertEqual(batch.spec_info.future_indices.tolist(), [0])
        s.tree_cache.free_kv_row.assert_not_called()
        s.token_to_kv_pool_allocator.free.assert_not_called()

    def test_other_algorithms_still_rejected_with_name(self):
        for algo in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.STANDALONE):
            with self.subTest(algo=algo):
                a = make_req("a", output_len=6, lookahead=4)
                s = self.make_scheduler([a])
                s.spec_algorithm = algo
                for out in (
                    s.pause_request(PauseReqInput(rid="a")),
                    s.resume_request(ResumeReqInput(rid="a")),
                    s.watermark_request(WatermarkReqInput(rid="a", watermark=1)),
                ):
                    self.assertFalse(out.success)
                    self.assertIn(algo.name, out.message)
                    self.assertIn("not supported", out.message)
                # The budget never pauses either.
                self.assertEqual([r.rid for r in self.step(s).reqs], ["a"])
                self.assertEqual(s.paused_reqs, {})

    def test_flag_off_rejects_under_ngram(self):
        a = make_req("a", output_len=6, lookahead=4)
        s = self.make_scheduler([a])
        s.enable_joint_checkpoint = False
        out = s.pause_request(PauseReqInput(rid="a"))
        self.assertFalse(out.success)
        self.assertIn("--enable-joint-checkpoint", out.message)

    # ------------------------------------------------------------ timeout

    def test_pause_timeout_aborts_and_resumes_under_ngram(self):
        a = make_req("a", output_len=6, lookahead=4, row=0)
        b = make_req("b", output_len=6, row=1)
        s = self.make_scheduler([a, b], timeout_s=10.0)
        self.seed_match_states(s, [a, b])
        clock = "sglang.srt.managers.scheduler.time.monotonic"
        with patch(clock, return_value=1000.0):
            self.step(s)
        self.assertNotIn("a", self.match_rids(s))
        with patch(clock, return_value=1009.0):
            self.assertEqual([r.rid for r in self.step(s).reqs], ["b"])
        spy = self.spy_resume_builder(s)
        with patch(clock, return_value=1010.5):
            batch = self.step(s)
        spy.assert_called_once_with([a])
        self.assertEqual([r.rid for r in batch.reqs], ["b", "a"])
        self.assertIsInstance(a.to_finish, FINISH_ABORT)
        self.assertIn("pause timeout", a.to_finish.message)
        self.assertEqual(s.paused_reqs, {})
        self.assertEqual(batch.spec_info.future_indices.tolist(), [1, 0])
        # Its decode step finishes it; the next filter drops it and the
        # batch's spec input follows.
        a.finished_reason = a.to_finish
        batch.filter_batch()
        self.assertEqual([r.rid for r in batch.reqs], ["b"])
        self.assertEqual(batch.spec_info.future_indices.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
