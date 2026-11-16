"""Joint checkpoint P3 (truncate) on the radix tree, CPU only: a Scheduler
built with Scheduler.__new__ over a real UnifiedRadixCache, real pools and
real Req objects.

- In flight (TestJointCheckpointTruncate): Scheduler._jc_truncate checks the
  successor's first pos tokens against the running (or paused) request's
  ledger, writes its KV [0, pos) into the tree, locks that node for the
  successor (a pin_table hold keyed by the successor's rid, released at its
  admission or abort, or by a short TTL), and finishes the request in place:
  output_ids cut at pos, tail KV freed, one final chunk streamed. The
  successor's prefix match then reaches pos, so only the appended tokens are
  extended.
- Finished: accepted only while a /pin_request hold on the request covers pos.
- Fork: children of a pin match the fork point; truncating one leaves the
  other alone.
- Rejections leave the request, the tree and the pool untouched.
- Overlap: a late result for the truncated request streams nothing more and
  frees nothing more.
- Admission (TestJointCheckpointAdmitFirstExclusive, pass-through
  PrefillAdder): successors within --jc-admit-first-max-tokens (estimated as
  input length past the truncate position) get a prefill batch of their own,
  without cold requests or the chunked request's next chunk, at most
  JC_ADMIT_FIRST_MAX_CONSECUTIVE times in a row; admission drops their
  truncate holds; flag off is admission as without the mark.
- Pin-on-finish (TestJointCheckpointPinOnFinish): a request carrying
  jc_pin_on_finish gets, when release_kv_cache writes its KV into the tree,
  the hold /pin_request would make (pin_table record, TTL, cap), reported in
  its final chunk; discard and releases without a tree insert take none.
  /pin_request on a finished rid renews that hold's lease.
- Distinct accounting (TestJointCheckpointDistinctPinAccounting, Rust and
  Python tree cores): the pin cap counts each tree token held by any pin
  once, across node splits and in any release order.
- Pin pressure valve (TestJointCheckpointPinPressureValve*): with
  --jc-pin-overflow lru a pin over the cap degrades other pins, least
  recently touched first and only those whose release makes room; a pin that
  cannot fit is refused with nothing degraded; refuse mode is unchanged.
"""

import itertools
import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.api.v1.api_types import GenerateRequest, JsonContractError
from sglang.srt.managers.io_struct import (
    AbortReq,
    GenerateReqInput,
    PinOnFinishParams,
    PinReqInput,
    SessionParams,
    TruncateParams,
    UnpinReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import JcPin, Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POOL_SIZE = 256
PROMPT = list(range(100, 110))  # 10 prompt tokens
OUTPUT = list(range(200, 206))  # 6 generated tokens
FILL = PROMPT + OUTPUT


def make_cache(
    page_size: int = 1,
    pool_size: int = POOL_SIZE,
    max_context_len: int = 128,
    tree_core_backend=None,
) -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    if page_size == 1:
        allocator = TokenToKVPoolAllocator(
            size=pool_size,
            dtype=dtype,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
    else:
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size,
            page_size=page_size,
            dtype=dtype,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
            tree_core_backend=tree_core_backend,
        )
    )


def make_req(rid, input_ids, extra_key=None) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", input_ids),
        sampling_params=SamplingParams(max_new_tokens=100),
        extra_key=extra_key,
    )


def recv_for(rid, old_rid, pos, session_params=None):
    return SimpleNamespace(
        rid=rid,
        truncate=TruncateParams(rid=old_rid, pos=pos),
        session_params=session_params,
    )


class _TruncateFixture(CustomTestCase):
    page_size = 1
    pool_size = POOL_SIZE
    max_context_len = 128
    tree_core_backend = None  # the default (Rust when available)

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.cache = make_cache(
            self.page_size,
            pool_size=self.pool_size,
            max_context_len=self.max_context_len,
            tree_core_backend=self.tree_core_backend,
        )
        self.allocator = self.cache.token_to_kv_pool_allocator
        self.now = 1000.0
        clock = SimpleNamespace(monotonic=lambda: self.now)
        patcher = patch.object(scheduler_module, "time", clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ------------------------------------------------------------ fixtures

    def make_scheduler(
        self, running=(), enable=True, streamer=None, max_ratio=0.5, overflow="refuse"
    ):
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.disaggregation_mode = DisaggregationMode.NULL
        s.dllm_config = None
        s.page_size = self.page_size
        s.pin_table = {}
        s.jc_pin_ttl_s = 600.0
        s.jc_pin_max_ttl_s = 3600.0
        s.jc_pin_max_ratio = max_ratio
        s.jc_pin_overflow = overflow
        s.jc_degraded_pins = {}
        s.jc_pins_degraded_total = 0
        s.jc_pins_repinned_total = 0
        s.max_total_num_tokens = self.pool_size
        s.paused_reqs = {}
        s.jc_pause_timeout_s = 1e9
        s.chunked_req = None
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.tree_cache = self.cache
        s.req_to_token_pool = self.cache.req_to_token_pool
        s.token_to_kv_pool_allocator = self.allocator
        s.running_batch = ScheduleBatch(reqs=list(running), batch_is_full=False)
        s.last_batch = None
        s.result_queue = deque()
        s.enable_continuous_input_polling = False
        s.waiting_queue = []
        s.draft_worker = None
        s.output_streamer = streamer if streamer is not None else MagicMock()
        return s

    def start(self, rid, prompt, outputs, overlap=True, extra_key=None) -> Req:
        """A request admitted (prefix matched and locked), prefilled and
        decoded ``outputs``. Under overlap the decode step for the last output
        token is in flight, so its KV slot is already allocated and committed
        (kv_committed_len == len(prompt + outputs)); without overlap that
        token has no KV yet (one less)."""
        req = make_req(rid, prompt, extra_key=extra_key)
        req.init_next_round_input(self.cache)
        req.lock = self.cache.lock(req.last_node)
        prefix = req.prefix_len
        total = len(prompt) + len(outputs)
        committed = total if overlap else total - 1
        pool = self.cache.req_to_token_pool
        row = pool.alloc_rows(1)[0]
        if prefix:
            pool.write(
                (row, slice(0, prefix)),
                self.cache.path_device_indices(req.last_node)[:prefix],
            )
        need = committed - prefix
        pages = -(need // -self.page_size)
        indices = self.allocator.alloc(pages * self.page_size)[:need]
        pool.write((row, slice(prefix, committed)), indices)
        req.kv.req_pool_idx = row
        req.kv.kv_committed_len = committed
        req.kv.kv_allocated_len = committed
        req.output_ids = array("q", outputs)
        req.refresh_fill_ids()
        return req

    def match_len(self, token_ids) -> int:
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        ).device_prefix_len

    def evict_all(self):
        self.cache.evict(EvictParams(num_tokens=self.pool_size))

    def truncate(self, s, old_rid, pos, append=(900, 901), prefix=None, rid="succ"):
        """Run Scheduler._jc_truncate for a successor whose input is
        ``prefix`` (default FILL[:pos]) + ``append``."""
        prefix = FILL[:pos] if prefix is None else prefix
        succ = make_req(rid, list(prefix) + list(append))
        return succ, s._jc_truncate(succ, recv_for(rid, old_rid, pos))

    def admit(self, s, succ):
        """What admission does for the successor: prefix match, own lock
        (PrefillAdder._commit_prefill_admission), then the hook in
        _get_new_batch_prefill_raw drops its truncate hold."""
        succ.init_next_round_input(self.cache)
        succ.lock = self.cache.lock(succ.last_node)
        s._jc_release_truncate_hold(succ, "admitted")

    def assert_untouched(self, s, old, available, n_out=len(OUTPUT)):
        self.assertFalse(old.finished())
        self.assertEqual(len(old.output_ids), n_out)
        self.assertTrue(old.kv.holds_kv)
        self.assertEqual(old.kv.cache_protected_len, 0)
        self.assertFalse(old.skip_radix_cache_insert)
        self.assertEqual(s.pin_table, {})
        self.assertEqual(self.allocator.available_size(), available)
        self.assertEqual(self.match_len(FILL), 0)
        s.output_streamer.stream_output.assert_not_called()

    def rejected(self, s, old, err, needle, available):
        self.assertIsNotNone(err)
        self.assertIn(needle, err)
        self.assert_untouched(s, old, available)

    def finished_with_pin(self, up_to):
        """``old`` pinned at up_to, then finished normally (its whole KV
        written to the tree) and filtered out; everything unpinned evicted."""
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        if up_to is not None:
            self.assertTrue(s.pin_request(PinReqInput(rid="old", up_to=up_to)).success)
        old.finished_reason = FINISH_LENGTH(length=len(OUTPUT))
        old.finished_len = len(OUTPUT)
        release_kv_cache(old, self.cache, checkpoint=True)
        s.running_batch.filter_batch()
        self.evict_all()
        return s

    def real_streamer(self) -> SchedulerOutputStreamer:
        # The streamer reads this process's ranks; none are built in a unit test.
        cm = get_parallel().override(attn_tp_rank=0, tp_rank=0, dp_rank=0)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return SchedulerOutputStreamer(
            send_to_detokenizer=MagicMock(),
            tree_cache=self.cache,
            server_args=ServerArgs(model_path="dummy"),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )


class TestJointCheckpointTruncate(_TruncateFixture):
    # ------------------------------------------------------------ the cut

    def test_truncate_running_request(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        pos = len(PROMPT) + 3
        available = self.allocator.available_size()
        rows = self.cache.req_to_token_pool.available_size()

        succ, err = self.truncate(s, "old", pos, append=(900, 901))
        self.assertIsNone(err)

        # The old request: finished without a forward, cut at pos.
        self.assertTrue(old.finished())
        self.assertIsInstance(old.finished_reason, FINISH_ABORT)
        self.assertEqual(
            old.finished_reason.message,
            f"Truncated to position {pos} by joint-checkpoint truncate.",
        )
        self.assertEqual(old.jc_truncate_to, pos)
        self.assertEqual(old.finished_len, 3)
        self.assertEqual(list(old.output_ids), OUTPUT[:3])
        self.assertEqual(list(old.full_untruncated_fill_ids), FILL[:pos])
        self.assertTrue(old.skip_radix_cache_insert)
        s.output_streamer.stream_output.assert_called_once()
        self.assertEqual(s.output_streamer.stream_output.call_args.args[0], [old])
        # Its row and the KV past pos are back in the pools; [0, pos) is the tree's.
        self.assertFalse(old.kv.holds_kv)
        self.assertIsNone(old.lock)
        self.assertEqual(self.cache.req_to_token_pool.available_size(), rows + 1)
        self.assertEqual(
            self.allocator.available_size(), available + len(FILL) - pos
        )
        self.assertEqual(self.match_len(FILL), pos)

        # The hold keeps [0, pos) for the successor, keyed by its rid.
        hold = s.pin_table["succ"]
        self.assertEqual(hold.successor_of, "old")
        self.assertEqual(hold.pinned_len, pos)
        self.assertEqual(hold.expires_at, self.now + Scheduler.JC_TRUNCATE_HOLD_TTL_S)
        self.assertEqual(self.cache.evictable_size(), 0)
        self.evict_all()
        self.assertEqual(self.match_len(FILL), pos)

        # The successor: admit-first, matches pos, extends only the append.
        self.assertTrue(succ.jc_admit_first)
        self.assertEqual(succ.jc_truncate_pos, pos)
        self.assertEqual(s._jc_appended_tokens(succ), 2)
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, pos)
        self.assertEqual(len(succ.full_untruncated_fill_ids) - succ.prefix_len, 2)
        # Admitted: the hold is gone, its own lock keeps the prefix.
        self.assertEqual(s.pin_table, {})
        self.evict_all()
        self.assertEqual(self.match_len(FILL), pos)
        self.cache.unlock(succ.lock)
        self.evict_all()
        self.assertEqual(self.match_len(FILL), 0)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

        # The next step's filter drops the cut request from its batch.
        s.running_batch.filter_batch()
        self.assertEqual(s.running_batch.reqs, [])

    def test_truncate_keeps_everything_and_appends(self):
        # pos == input + output: the in-flight decode step's slot is the last
        # kept position (committed), nothing is freed.
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        succ, err = self.truncate(s, "old", len(FILL), append=(900,))
        self.assertIsNone(err)
        self.assertEqual(list(old.output_ids), OUTPUT)
        self.assertEqual(self.allocator.available_size(), available)
        self.assertEqual(s.pin_table["succ"].pinned_len, len(FILL))
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, len(FILL))
        self.assertEqual(len(succ.full_untruncated_fill_ids) - succ.prefix_len, 1)

    def test_truncate_without_overlap_keeps_committed_kv_only(self):
        # Without overlap the last output token has no KV yet: the tree gets
        # [0, total - 1) and the successor recomputes that one token.
        old = self.start("old", PROMPT, OUTPUT, overlap=False)
        s = self.make_scheduler([old])
        succ, err = self.truncate(s, "old", len(FILL), append=(900,))
        self.assertIsNone(err)
        self.assertEqual(list(old.output_ids), OUTPUT)
        self.assertEqual(s.pin_table["succ"].pinned_len, len(FILL) - 1)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE - (len(FILL) - 1))
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, len(FILL) - 1)
        self.assertEqual(len(succ.full_untruncated_fill_ids) - succ.prefix_len, 2)

    def test_truncate_at_input_len_keeps_prompt_only(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        succ, err = self.truncate(s, "old", len(PROMPT), append=(900,))
        self.assertIsNone(err)
        self.assertEqual(list(old.output_ids), [])
        self.assertEqual(self.match_len(FILL), len(PROMPT))
        self.assertEqual(self.allocator.available_size(), POOL_SIZE - len(PROMPT))
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, len(PROMPT))

    def test_truncate_below_a_pin_keeps_the_pinned_kv(self):
        # A pin published [0, 15) already; cutting at 12 must not free the
        # tree-owned [12, 15): only [15, 16) goes back to the pool.
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        out = s.pin_request(PinReqInput(rid="old", up_to=15))
        self.assertTrue(out.success)
        pos = 12
        succ, err = self.truncate(s, "old", pos, append=(900,))
        self.assertIsNone(err)
        self.assertEqual(list(old.output_ids), OUTPUT[:2])
        self.assertEqual(old.finished_len, 2)
        self.assertEqual(s.pin_table["succ"].pinned_len, 15)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE - 15)
        self.assertEqual(self.match_len(FILL), 15)
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, pos)
        self.cache.unlock(succ.lock)
        s.unpin_request(SimpleNamespace(rid="old"))
        self.evict_all()
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_truncate_paused_request(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([])
        old.jc_pause_reason = "manual"
        s.paused_reqs[old.rid] = old
        succ, err = self.truncate(s, "old", len(PROMPT) + 1)
        self.assertIsNone(err)
        self.assertTrue(old.finished())
        self.assertIn("succ", s.pin_table)
        # The pause bookkeeping drops it as finished at the next step.
        s._jc_on_step(s.running_batch)
        self.assertEqual(s.paused_reqs, {})

    def test_draft_worker_is_told(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        s.draft_worker = MagicMock()
        _, err = self.truncate(s, "old", len(PROMPT) + 1)
        self.assertIsNone(err)
        s.draft_worker.note_request_finished.assert_called_once_with(
            rid="old", natural_stop=False
        )

    # ------------------------------------------------------------ the hold

    def test_hold_released_when_successor_aborted_while_waiting(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        pos = len(PROMPT) + 3
        succ, err = self.truncate(s, "old", pos)
        self.assertIsNone(err)
        s.waiting_queue = [succ]
        s.mm_receiver = None
        s._release_aborted_request = lambda req: None
        s.beam_coordinator = SimpleNamespace(retire_group=lambda req: None)
        s.send_abort_output = MagicMock()
        s.grammar_manager = MagicMock()
        s.abort_request(AbortReq(rid="succ"))
        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(s.pin_table, {})
        self.assertEqual(self.cache.evictable_size(), pos)
        self.evict_all()
        self.assertEqual(self.match_len(FILL), 0)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_hold_expires_after_ttl(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        succ, err = self.truncate(s, "old", len(PROMPT) + 3)
        self.assertIsNone(err)
        self.now += Scheduler.JC_TRUNCATE_HOLD_TTL_S - 1
        s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))
        self.assertIn("succ", s.pin_table)
        self.now += 1
        s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))
        self.assertEqual(s.pin_table, {})
        self.evict_all()
        self.assertEqual(self.match_len(FILL), 0)
        # A later admission of the successor finds no hold to drop.
        s._jc_release_truncate_hold(succ, "admitted")
        self.assertEqual(s.pin_table, {})

    def test_admission_leaves_client_pins_alone(self):
        s = self.make_scheduler([])
        sentinel = JcPin(lock=None, pinned_len=4, pinned_at=0.0, expires_at=None)
        s.pin_table["succ"] = sentinel
        s.tree_cache = MagicMock()
        s._jc_release_truncate_hold(make_req("succ", [1, 2]), "admitted")
        self.assertIs(s.pin_table["succ"], sentinel)
        s.tree_cache.unlock.assert_not_called()

    def test_truncate_succeeds_without_hold_over_pin_cap(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old], max_ratio=0.01)  # limit 2 tokens
        pos = len(PROMPT) + 3
        with self.assertLogs(scheduler_module.logger, level="INFO") as logs:
            succ, err = self.truncate(s, "old", pos)
        self.assertIsNone(err)
        self.assertTrue(old.finished())
        self.assertTrue(succ.jc_admit_first)
        self.assertEqual(s.pin_table, {})
        self.assertTrue(any("takes no hold" in m for m in logs.output))
        # In the tree but not held: evictable until the successor locks it.
        self.assertEqual(self.match_len(FILL), pos)
        self.assertEqual(self.cache.evictable_size(), pos)

    # ------------------------------------------------------------ rejections

    def test_rejected_without_flag(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old], enable=False)
        available = self.allocator.available_size()
        succ, err = self.truncate(s, "old", len(PROMPT) + 2)
        self.rejected(s, old, err, "requires --enable-joint-checkpoint", available)
        self.assertFalse(succ.jc_admit_first)

    def test_rejected_prefix_mismatch(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        pos = len(PROMPT) + 3
        prefix = list(FILL[:pos])
        prefix[11] = 999
        succ, err = self.truncate(s, "old", pos, prefix=prefix)
        self.rejected(
            s, old, err, "input_ids differ from request old at position 11", available
        )
        self.assertFalse(succ.jc_admit_first)
        self.assertIsNone(succ.jc_truncate_pos)

    def test_rejected_input_shorter_than_pos(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        succ, err = self.truncate(s, "old", 14, prefix=FILL[:12], append=())
        self.rejected(s, old, err, "shorter than truncate pos=14", available)

    def test_rejected_pos_out_of_range(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        _, err = self.truncate(s, "old", len(PROMPT) - 1)
        self.rejected(s, old, err, "outside [10, 16]", available)
        _, err = self.truncate(s, "old", len(FILL) + 1, prefix=FILL + [0])
        self.rejected(s, old, err, "outside [10, 16]", available)

    def test_rejected_while_prefilling_or_finishing(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        s.chunked_req = old
        _, err = self.truncate(s, "old", len(PROMPT) + 1)
        self.rejected(s, old, err, "request old has not finished prefill", available)
        s.chunked_req = None
        old.to_finish = FINISH_ABORT()
        _, err = self.truncate(s, "old", len(PROMPT) + 1)
        self.rejected(s, old, err, "request old is finishing", available)
        old.to_finish = None
        # Not admitted yet: still in the waiting queue.
        waiting = make_req("queued", PROMPT)
        s.waiting_queue = [waiting]
        _, err = self.truncate(s, "queued", len(PROMPT))
        self.assertEqual(err, "request queued has not finished prefill")

    def test_rejected_other_namespace_session_or_barred(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        pos = len(PROMPT) + 1
        succ = make_req("succ", FILL[:pos] + [900], extra_key="other")
        err = s._jc_truncate(succ, recv_for("succ", "old", pos))
        self.rejected(s, old, err, "cache namespace", available)

        succ = make_req("succ", FILL[:pos] + [900])
        err = s._jc_truncate(
            succ, recv_for("succ", "old", pos, session_params=SessionParams(id="x"))
        )
        self.rejected(s, old, err, "cannot be combined with session_params", available)

        old.session = SimpleNamespace(streaming=True)
        _, err = self.truncate(s, "old", pos)
        self.rejected(s, old, err, "streaming-session", available)
        old.session = None

        old.skip_radix_cache_insert = True
        _, err = self.truncate(s, "old", pos)
        self.assertIn("barred from the radix tree", err)
        old.skip_radix_cache_insert = False
        self.assert_untouched(s, old, available)

    def test_rejected_on_disaggregation(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        s.disaggregation_mode = DisaggregationMode.PREFILL
        available = self.allocator.available_size()
        _, err = self.truncate(s, "old", len(PROMPT) + 1)
        self.rejected(s, old, err, "disaggregation", available)

    # ------------------------------------------------------------ finished request

    def test_finished_request_with_covering_pin(self):
        s = self.finished_with_pin(up_to=13)
        pin = s.pin_table["old"]
        for pos in (len(PROMPT), 12, 13):
            succ, err = self.truncate(s, "old", pos, rid=f"succ{pos}")
            self.assertIsNone(err, pos)
            self.assertTrue(succ.jc_admit_first)
            self.assertEqual(succ.jc_truncate_pos, pos)
            # No hold of its own: the pin keeps the prefix, untouched.
            self.assertEqual(set(s.pin_table), {"old"})
            self.assertIs(s.pin_table["old"], pin)
            self.admit(s, succ)
            self.assertEqual(succ.prefix_len, pos)
            self.cache.unlock(succ.lock)
        s.output_streamer.stream_output.assert_not_called()

    def test_finished_request_rejected_without_covering_pin(self):
        msg = "request old is not running and no pin covers position {}"
        s = self.finished_with_pin(up_to=13)
        _, err = self.truncate(s, "old", 14)
        self.assertEqual(err, msg.format(14))
        # Expired but not yet stepped: no longer counts.
        self.now = s.pin_table["old"].expires_at
        _, err = self.truncate(s, "old", 12)
        self.assertEqual(err, msg.format(12))
        self.assertIn("old", s.pin_table)

        s = self.finished_with_pin(up_to=None)
        _, err = self.truncate(s, "old", 12)
        self.assertEqual(err, msg.format(12))
        _, err = self.truncate(s, "never-seen", 12)
        self.assertIn("request never-seen is not running", err)

    def test_finished_request_still_in_batch_needs_a_pin(self):
        # Finished by the last result, not yet filtered: its KV is gone.
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        old.finished_reason = FINISH_LENGTH(length=len(OUTPUT))
        old.finished_len = len(OUTPUT)
        release_kv_cache(old, self.cache, checkpoint=True)
        _, err = self.truncate(s, "old", 12)
        self.assertEqual(err, "request old is not running and no pin covers position 12")

    def test_a_truncate_hold_does_not_count_as_a_pin(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        succ, err = self.truncate(s, "old", 13, rid="succ")
        self.assertIsNone(err)
        # "succ" never ran; its hold is not a pin on it.
        _, err = self.truncate(s, "succ", 12, rid="succ2")
        self.assertIn("no pin covers position 12", err)

    # ------------------------------------------------------------ fork

    def test_fork_children_share_the_pin_and_truncate_independently(self):
        parent = self.start("parent", PROMPT, OUTPUT)
        s = self.make_scheduler([parent])
        fork_pos = 12
        self.assertTrue(s.pin_request(PinReqInput(rid="parent", up_to=fork_pos)).success)

        a = self.start("a", FILL[:fork_pos] + [700, 701], [710, 711, 712])
        b = self.start("b", FILL[:fork_pos] + [800], [810, 811])
        self.assertEqual(a.prefix_len, fork_pos)
        self.assertEqual(b.prefix_len, fork_pos)
        s.running_batch = ScheduleBatch(reqs=[parent, a, b], batch_is_full=False)
        b_state = (
            list(b.output_ids),
            b.kv.req_pool_idx,
            b.kv.kv_committed_len,
            b.kv.cache_protected_len,
            b.lock,
        )
        b_row = self.cache.req_to_token_pool.req_to_token[b.kv.req_pool_idx].clone()

        a_fill = list(a.full_untruncated_fill_ids)
        pos = len(a_fill) - 1  # a's prompt + 2 of its 3 outputs
        succ, err = self.truncate(s, "a", pos, prefix=a_fill[:pos], append=(990,))
        self.assertIsNone(err)
        self.assertTrue(a.finished())

        # b is untouched: same ledger, row, lock and row contents.
        self.assertFalse(b.finished())
        self.assertEqual(
            (
                list(b.output_ids),
                b.kv.req_pool_idx,
                b.kv.kv_committed_len,
                b.kv.cache_protected_len,
                b.lock,
            ),
            b_state,
        )
        self.assertTrue(
            torch.equal(
                self.cache.req_to_token_pool.req_to_token[b.kv.req_pool_idx], b_row
            )
        )
        # The parent's pin is untouched; a's continuation matches pos.
        self.assertEqual(s.pin_table["parent"].pinned_len, fork_pos)
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, pos)
        self.evict_all()
        self.assertEqual(self.match_len(FILL), fork_pos)
        self.assertEqual(self.match_len(list(b.full_untruncated_fill_ids)), fork_pos)

    # ------------------------------------------------------------ overlap

    def test_late_result_for_truncated_request_streams_and_frees_nothing(self):
        old = self.start("old", PROMPT, OUTPUT)
        streamer = self.real_streamer()
        s = self.make_scheduler([old], streamer=streamer)
        pos = len(PROMPT) + 2
        _, err = self.truncate(s, "old", pos)
        self.assertIsNone(err)

        # Exactly one payload: the final chunk, abort / truncated.
        send = streamer.send_to_detokenizer.send_output
        self.assertEqual(send.call_count, 1)
        payload = send.call_args.args[0]
        self.assertEqual(payload.rids, ["old"])
        self.assertEqual(payload.finished_reasons[0]["type"], "abort")
        self.assertTrue(payload.finished_reasons[0]["message"].startswith("Truncated"))
        self.assertTrue(old.finished_output)

        available = self.allocator.available_size()
        rows = self.cache.req_to_token_pool.available_size()
        # The in-flight decode result lands: process_batch_result_decode skips
        # a finished request (no output append, no release) and streams the
        # batch; the streamer skips a request whose final chunk went out.
        self.assertTrue(old.finished())
        streamer.stream_output([old], False)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(list(old.output_ids), OUTPUT[:2])
        self.assertEqual(self.allocator.available_size(), available)
        self.assertEqual(self.cache.req_to_token_pool.available_size(), rows)
        self.assertEqual(self.match_len(FILL), pos)


class TestJointCheckpointTruncatePaged(_TruncateFixture):
    """page_size 4: the tree keeps whole pages below pos, the successor
    recomputes the partial page, pin coverage is judged page-aligned."""

    page_size = 4

    def test_rejected_prefix_mismatch(self):
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old])
        available = self.allocator.available_size()
        prefix = list(FILL[:14])
        prefix[13] = 999
        _, err = self.truncate(s, "old", 14, prefix=prefix)
        self.rejected(
            s, old, err, "input_ids differ from request old at position 13", available
        )

    def test_late_result_for_truncated_request_streams_and_frees_nothing(self):
        # pos 12 is page-aligned: the same assertions hold.
        old = self.start("old", PROMPT, OUTPUT)
        streamer = self.real_streamer()
        s = self.make_scheduler([old], streamer=streamer)
        _, err = self.truncate(s, "old", 12)
        self.assertIsNone(err)
        send = streamer.send_to_detokenizer.send_output
        self.assertEqual(send.call_count, 1)
        available = self.allocator.available_size()
        streamer.stream_output([old], False)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.allocator.available_size(), available)
        self.assertEqual(self.match_len(FILL), 12)

    def test_paged_truncate_keeps_whole_pages(self):
        old = self.start("old", PROMPT, OUTPUT)  # 16 tokens = 4 pages
        s = self.make_scheduler([old])
        pos = 14
        succ, err = self.truncate(s, "old", pos, append=(900,))
        self.assertIsNone(err)
        # [0, 12) in the tree; the page holding 12..13 and the last page freed.
        self.assertEqual(s.pin_table["succ"].pinned_len, 12)
        self.assertEqual(self.match_len(FILL), 12)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE - 12)
        self.assertEqual(list(old.output_ids), OUTPUT[:4])
        # Estimate = real extend: 15 tokens, match stops at page 12.
        self.assertEqual(s._jc_appended_tokens(succ), 3)
        self.admit(s, succ)
        self.assertEqual(succ.prefix_len, 12)
        self.assertEqual(len(succ.full_untruncated_fill_ids) - succ.prefix_len, 3)
        self.assertEqual(s.pin_table, {})

    def test_paged_pin_coverage_is_page_aligned(self):
        s = self.finished_with_pin(up_to=13)  # pins 12 = page floor
        self.assertEqual(s.pin_table["old"].pinned_len, 12)
        for pos in (12, 13, 15):
            succ, err = self.truncate(s, "old", pos, rid=f"s{pos}")
            self.assertIsNone(err, pos)
            self.admit(s, succ)
            self.assertEqual(succ.prefix_len, 12)
            self.cache.unlock(succ.lock)
        _, err = self.truncate(s, "old", 16, prefix=FILL, append=(1,))
        self.assertIn("no pin covers position 16", err)


class _PinOnFinishFixture(_TruncateFixture):
    def armed(self, s, rid, prompt=PROMPT, outputs=OUTPUT, ttl_s=None, **kw) -> Req:
        """A running request whose /generate carried jc_pin_on_finish."""
        req = self.start(rid, prompt, outputs, **kw)
        recv = SimpleNamespace(
            session_params=None, jc_pin_on_finish=PinOnFinishParams(ttl_s=ttl_s)
        )
        self.assertIsNone(s._jc_arm_pin_on_finish(req, recv))
        s.running_batch.reqs = s.running_batch.reqs + [req]
        return req

    def finish(self, req, reason=None, checkpoint=True):
        """What process_batch_result does once the request finishes."""
        req.finished_reason = reason or FINISH_LENGTH(length=len(req.output_ids))
        req.finished_len = len(req.output_ids)
        release_kv_cache(req, self.cache, checkpoint=checkpoint)

    def abort(self, s, rid, discard=False):
        s.mm_receiver = None
        s.grammar_manager = MagicMock()
        s.abort_request(AbortReq(rid=rid, discard=discard))
        req = next(r for r in s.running_batch.reqs if r.rid == rid)
        self.finish(req, reason=req.to_finish)
        return req


class TestJointCheckpointPinOnFinish(_PinOnFinishFixture):
    def test_finish_holds_the_whole_kv_like_a_pin(self):
        streamer = self.real_streamer()
        s = self.make_scheduler(streamer=streamer)
        old = self.armed(s, "old")
        self.finish(old)
        pin = s.pin_table["old"]
        self.assertEqual(
            (pin.pinned_len, pin.pinned_at, pin.expires_at, pin.successor_of),
            (len(FILL), 1000.0, 1600.0, None),
        )
        self.evict_all()
        self.assertEqual(self.match_len(FILL), len(FILL))
        streamer.stream_output([old], False)
        payload = streamer.send_to_detokenizer.send_output.call_args[0][0]
        self.assertEqual(payload.jc_pin_outcomes, [{"jc_pinned_len": len(FILL)}])
        # Without the field nothing extra goes on the wire.
        plain = self.start("plain", [p + 1000 for p in PROMPT], OUTPUT)
        self.finish(plain)
        streamer.stream_output([plain], False)
        payload = streamer.send_to_detokenizer.send_output.call_args[0][0]
        self.assertIsNone(payload.jc_pin_outcomes)
        self.assertEqual(set(s.pin_table), {"old"})

    def test_plain_abort_holds_discard_does_not(self):
        s = self.make_scheduler()
        other = [p + 1000 for p in PROMPT]
        aborted = self.armed(s, "a")
        discarded = self.armed(s, "d", prompt=other)
        self.abort(s, "a")
        self.abort(s, "d", discard=True)
        self.assertEqual(set(s.pin_table), {"a"})
        self.assertEqual(aborted.jc_pin_outcome, {"jc_pinned_len": len(FILL)})
        self.assertEqual(
            discarded.jc_pin_outcome,
            {
                "jc_pinned_len": 0,
                "jc_pin_message": "request finished without writing its KV to the radix tree",
            },
        )
        self.evict_all()
        self.assertEqual(self.match_len(FILL), len(FILL))
        self.assertEqual(self.match_len(other + OUTPUT), 0)

    def test_release_without_tree_insert_keeps_the_hook_armed(self):
        """A release that writes nothing to the tree (retraction, chunked
        abort) takes no hold and must not disarm a later finish."""
        s = self.make_scheduler()
        old = self.armed(s, "old")
        release_kv_cache(old, self.cache, checkpoint=False)
        self.assertEqual(s.pin_table, {})
        self.assertIsNotNone(old.jc_finish_hold)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_cap_refusal_takes_no_hold_and_keeps_an_earlier_pin(self):
        # limit int(0.12 x 256) = 30: "b" holds 16, "old" its own 12.
        s = self.make_scheduler(max_ratio=0.12)
        b = self.armed(s, "b", prompt=[p + 1000 for p in PROMPT])
        self.finish(b)
        old = self.armed(s, "old")
        self.assertTrue(s.pin_request(PinReqInput(rid="old", up_to=12)).success)
        earlier = s.pin_table["old"]
        self.finish(old)
        self.assertIs(s.pin_table["old"], earlier)
        self.assertEqual(old.jc_pin_outcome["jc_pinned_len"], 0)
        self.assertEqual(
            old.jc_pin_outcome["jc_pin_message"],
            "pin cap exceeded: 32 distinct pinned tokens with this hold on old "
            "(16 without it) > limit 30 (--jc-pin-max-ratio 0.12 x 256)",
        )
        self.evict_all()
        self.assertEqual(self.match_len(FILL), 12)

    def test_earlier_client_pin_is_replaced_without_leaking_its_lock(self):
        # limit 25: 12 (earlier pin) + 16 would be refused unless the rid's
        # own earlier hold is replaced, as on a re-pin.
        s = self.make_scheduler(max_ratio=0.1)
        old = self.armed(s, "old")
        self.assertTrue(s.pin_request(PinReqInput(rid="old", up_to=12)).success)
        self.finish(old)
        self.assertEqual(s.pin_table["old"].pinned_len, len(FILL))
        self.assertTrue(s.unpin_request(UnpinReqInput(rid="old")).success)
        self.evict_all()
        self.assertEqual(self.match_len(FILL), 0)
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_ttl_expiry_and_unpin_release_the_hold(self):
        s = self.make_scheduler()
        short = self.armed(s, "short", ttl_s=5.0)
        kept = self.armed(s, "kept", prompt=[p + 1000 for p in PROMPT], ttl_s=3600)
        self.finish(short)
        self.finish(kept)
        self.assertEqual(s.pin_table["short"].expires_at, 1005.0)
        self.assertEqual(s.pin_table["kept"].expires_at, 4600.0)
        self.now += 5
        s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))
        self.assertEqual(set(s.pin_table), {"kept"})
        s.unpin_request(UnpinReqInput(rid="kept"))
        self.evict_all()
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_truncate_of_the_finished_request_uses_the_hold(self):
        s = self.make_scheduler()
        old = self.armed(s, "old")
        self.finish(old)
        s.running_batch.filter_batch()
        self.evict_all()
        for pos in (len(PROMPT), len(FILL)):
            succ, err = self.truncate(s, "old", pos, rid=f"succ{pos}")
            self.assertIsNone(err, pos)
            self.admit(s, succ)
            self.assertEqual(succ.prefix_len, pos)
            self.cache.unlock(succ.lock)
        _, err = self.truncate(s, "old", len(FILL) + 1, prefix=FILL + [7])
        self.assertIn("no pin covers position", err)

    def test_flag_off_runs_unpinned_and_reports_it(self):
        s = self.make_scheduler(enable=False)
        old = self.start("old", PROMPT, OUTPUT)
        recv = SimpleNamespace(
            session_params=None, jc_pin_on_finish=PinOnFinishParams()
        )
        self.assertIsNone(s._jc_arm_pin_on_finish(old, recv))
        self.finish(old)
        self.assertEqual(s.pin_table, {})
        self.assertEqual(
            old.jc_pin_outcome,
            {
                "jc_pinned_len": 0,
                "jc_pin_message": "--enable-joint-checkpoint is not set",
            },
        )

    def test_rejected_with_session_params_or_when_barred(self):
        s = self.make_scheduler()
        req = make_req("r", PROMPT)
        recv = SimpleNamespace(
            session_params=SessionParams(id="x"), jc_pin_on_finish=PinOnFinishParams()
        )
        self.assertIn("session_params", s._jc_arm_pin_on_finish(req, recv))
        recv.session_params = None
        for ttl_s, needle in ((0, "ttl_s must be > 0"), (7200, "exceeds")):
            recv.jc_pin_on_finish = PinOnFinishParams(ttl_s=ttl_s)
            self.assertIn(needle, s._jc_arm_pin_on_finish(req, recv))
        recv.jc_pin_on_finish = PinOnFinishParams()
        req.skip_radix_cache_insert = True
        self.assertIn("barred", s._jc_arm_pin_on_finish(req, recv))
        self.assertIsNone(req.jc_finish_hold)

    def test_pin_request_renews_a_finished_hold(self):
        """A rid that is no longer running renews its hold's lease; coverage
        can shrink nothing and grow nothing."""
        s = self.make_scheduler()
        old = self.armed(s, "old")
        self.finish(old)
        s.running_batch.filter_batch()
        pin = s.pin_table["old"]
        self.now = 1500.0
        out = s.pin_request(PinReqInput(rid="old", ttl_s=100.0))
        self.assertTrue(out.success, out.message)
        self.assertEqual((out.pinned_len, out.expires_in_s), (len(FILL), 100.0))
        self.assertEqual((pin.pinned_at, pin.expires_at), (1500.0, 1600.0))
        out = s.pin_request(PinReqInput(rid="old", up_to=len(FILL) + 1))
        self.assertFalse(out.success)
        self.assertIn("cannot be extended to 17", out.message)
        self.assertEqual(pin.expires_at, 1600.0)
        out = s.pin_request(PinReqInput(rid="old", up_to=10))
        self.assertTrue(out.success)
        self.assertEqual(out.pinned_len, len(FILL))
        self.assertEqual(pin.expires_at, 2100.0)
        self.assertIs(s.pin_table["old"], pin)
        # One lock throughout: unpin leaves nothing held.
        s.unpin_request(UnpinReqInput(rid="old"))
        self.evict_all()
        self.assertEqual(self.allocator.available_size(), POOL_SIZE)

    def test_renewal_refused_for_truncate_holds_expired_pins_and_bad_ttl(self):
        s = self.make_scheduler()
        old = self.armed(s, "old")
        self.finish(old)
        s.running_batch.filter_batch()
        _, err = self.truncate(s, "old", 12)
        self.assertIsNone(err)
        out = s.pin_request(PinReqInput(rid="succ"))
        self.assertEqual(out.message, "request succ is not running")
        out = s.pin_request(PinReqInput(rid="old", ttl_s=0))
        self.assertEqual(out.message, "ttl_s must be > 0, got 0")
        # Expired, not yet released by the next step: gone for renewal too.
        self.now = s.pin_table["old"].expires_at
        out = s.pin_request(PinReqInput(rid="old"))
        self.assertEqual(out.message, "request old is not running")

    def test_truncate_hold_and_client_pin_on_one_path_count_once(self):
        # limit int(0.08 x 256) = 20: the successor hold of 16 on the path the
        # client pin already holds adds nothing, so it fits.
        old = self.start("old", PROMPT, OUTPUT)
        s = self.make_scheduler([old], max_ratio=0.08)
        self.assertTrue(s.pin_request(PinReqInput(rid="old", up_to=16)).success)
        _, err = self.truncate(s, "old", len(FILL), prefix=FILL, append=(900,))
        self.assertIsNone(err)
        self.assertEqual(s.pin_table["succ"].pinned_len, len(FILL))
        self.assertEqual(s._jc_pinned_tokens_total(), len(FILL))
        self.assertEqual(s._jc_pinned_tokens_sum(), 2 * len(FILL))


class TestJointCheckpointPinOnFinishPaged(_PinOnFinishFixture):
    page_size = 4

    def test_finish_hold_is_page_aligned(self):
        s = self.make_scheduler()
        # Without overlap the last token has no KV: 15 committed, 12 in pages.
        old = self.armed(s, "old", overlap=False)
        self.finish(old)
        self.assertEqual(s.pin_table["old"].pinned_len, 12)
        self.assertEqual(old.jc_pin_outcome, {"jc_pinned_len": 12})


class TestJointCheckpointPinOnFinishWire(CustomTestCase):
    def test_generate_schema_accepts_the_field(self):
        """The api_contract gate validates /generate against the generated
        schema; a field missing there is a 400 "unknown field"."""
        req = GenerateRequest.from_json_value(
            {"input_ids": [1, 2], "jc_pin_on_finish": {"ttl_s": 5}}
        )
        self.assertEqual(req.jc_pin_on_finish.ttl_s, 5.0)
        GenerateRequest.from_json_value({"input_ids": [1], "jc_pin_on_finish": {}})
        with self.assertRaises(JsonContractError):
            GenerateRequest.from_json_value(
                {"input_ids": [1], "jc_pin_on_finish": {"ttl": 5}}
            )

    def test_normalization_rejects_batches_and_bad_shapes(self):
        for bad in ({"ttl": 5}, {"ttl_s": "5"}, {"ttl_s": True}, True):
            obj = GenerateReqInput(input_ids=[1, 2], jc_pin_on_finish=bad)
            with self.assertRaises(ValueError, msg=bad):
                obj.normalize_batch_and_arguments()
        obj = GenerateReqInput(input_ids=[[1], [2]], jc_pin_on_finish={})
        with self.assertRaisesRegex(ValueError, "batch"):
            obj.normalize_batch_and_arguments()
        obj = GenerateReqInput(input_ids=[1, 2], jc_pin_on_finish={"ttl_s": None})
        obj.normalize_batch_and_arguments()


PREFIX_LEN, TAIL_LEN = 4300, 3700
SHARED_PREFIX = list(range(10_000, 10_000 + PREFIX_LEN))


def session_prompt(i):
    """The shared prefix and session i's own tail, less the one output token."""
    start = 100_000 * (i + 1)
    return SHARED_PREFIX + list(range(start, start + TAIL_LEN - 1))


class _SharedPrefixFixture(_PinOnFinishFixture):
    pool_size = 32768
    max_context_len = 8192

    def hold(self, s, i):
        req = self.armed(s, f"r{i}", prompt=session_prompt(i), outputs=[7])
        self.finish(req)
        return req


class TestJointCheckpointDistinctPinAccounting(_SharedPrefixFixture):
    """The pin cap counts the distinct tree tokens held by any pin: a prefix
    shared by several holds counts once, read off the tree so a node split
    under a held path changes nothing."""

    def test_shared_prefix_counted_once_across_a_split(self):
        s = self.make_scheduler()
        self.hold(s, 0)
        self.assertEqual(s._jc_pinned_tokens_total(), 8000)
        # Admitting r1 matches the shared prefix and splits r0's node there.
        r1 = self.armed(s, "r1", prompt=session_prompt(1), outputs=[7])
        self.assertEqual(r1.prefix_len, PREFIX_LEN)
        self.assertEqual(s._jc_pinned_tokens_total(), 8000)
        self.finish(r1)
        self.assertEqual(s._jc_pinned_tokens_total(), PREFIX_LEN + 2 * TAIL_LEN)
        self.hold(s, 2)
        out = s.pin_request(PinReqInput(rid="r2"))
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.pinned_tokens_total, PREFIX_LEN + 3 * TAIL_LEN)
        self.assertEqual(out.pinned_tokens_sum, 3 * 8000)

    def test_release_in_any_order_returns_to_zero(self):
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                self.setUp()
                s = self.make_scheduler()
                for i in range(3):
                    self.hold(s, i)
                for k, i in enumerate(order):
                    s.unpin_request(UnpinReqInput(rid=f"r{i}"))
                    left = 2 - k
                    want = PREFIX_LEN + left * TAIL_LEN if left else 0
                    self.assertEqual(s._jc_pinned_tokens_total(), want)
                self.evict_all()
                self.assertEqual(self.allocator.available_size(), self.pool_size)

    def test_cap_refusal_uses_the_distinct_count(self):
        # limit 16384: three holds (24000 summed, 15400 distinct) fit; a
        # fourth tail does not.
        s = self.make_scheduler(max_ratio=0.5)
        for i in range(3):
            self.assertEqual(self.hold(s, i).jc_pin_outcome, {"jc_pinned_len": 8000})
        fourth = self.hold(s, 3)
        self.assertEqual(
            fourth.jc_pin_outcome["jc_pin_message"],
            "pin cap exceeded: 19100 distinct pinned tokens with this hold on r3 "
            "(15400 without it) > limit 16384 (--jc-pin-max-ratio 0.5 x 32768)",
        )
        self.assertEqual(set(s.pin_table), {"r0", "r1", "r2"})


class TestJointCheckpointDistinctPinAccountingPythonCore(
    TestJointCheckpointDistinctPinAccounting
):
    tree_core_backend = "python"


class TestJointCheckpointPinPressureValve(_PinOnFinishFixture):
    """--jc-pin-overflow lru: a pin over --jc-pin-max-ratio degrades other
    pins, least recently pinned or renewed first, instead of being refused.
    Prompts are disjoint, so every 16-token hold counts in full."""

    def prompt(self, k):
        return [p + 1000 * (k + 1) for p in PROMPT]

    def held_at(self, s, rid, k, t):
        self.now = t
        req = self.armed(s, rid, prompt=self.prompt(k))
        self.finish(req)
        return req

    def three_then_a_fourth(self, overflow):
        """a, b, c finish with holds at t = 1000, 1001, 1002 under a 40-token
        limit (int(0.16 x 256)): c needs room."""
        s = self.make_scheduler(max_ratio=0.16, overflow=overflow)
        self.held_at(s, "a", 0, 1000.0)
        self.held_at(s, "b", 1, 1001.0)
        c = self.held_at(s, "c", 2, 1002.0)
        return s, c

    def test_degrades_least_recently_touched_first(self):
        s, c = self.three_then_a_fourth("lru")
        self.assertEqual(
            c.jc_pin_outcome, {"jc_pinned_len": 16, "jc_pin_degraded": ["a"]}
        )
        self.assertEqual(set(s.pin_table), {"b", "c"})
        # A renewal touches b, so c is now the least recently touched.
        self.now = 1003.0
        self.assertTrue(s.pin_request(PinReqInput(rid="b")).success)
        self.now = 1004.0
        d = self.start("d", self.prompt(3), OUTPUT)
        s.running_batch.reqs = s.running_batch.reqs + [d]
        out = s.pin_request(PinReqInput(rid="d"))
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.degraded, ["c"])
        self.assertEqual((out.pinned_tokens_total, out.pins_degraded_total), (32, 2))
        self.assertEqual(set(s.pin_table), {"b", "d"})
        # Degraded KV is not freed, only evictable.
        a_fill = self.prompt(0) + OUTPUT
        self.assertEqual(self.match_len(a_fill), 16)
        self.evict_all()
        self.assertEqual(self.match_len(a_fill), 0)
        self.assertEqual(self.match_len(self.prompt(1) + OUTPUT), 16)

    def test_refuse_mode_keeps_first_come_first_served(self):
        s, c = self.three_then_a_fourth("refuse")
        self.assertEqual(
            c.jc_pin_outcome,
            {
                "jc_pinned_len": 0,
                "jc_pin_message": "pin cap exceeded: 48 distinct pinned tokens "
                "with this hold on c (32 without it) > limit 40 "
                "(--jc-pin-max-ratio 0.16 x 256)",
            },
        )
        self.assertEqual(set(s.pin_table), {"a", "b"})
        self.assertEqual((s.jc_degraded_pins, s.jc_pins_degraded_total), ({}, 0))

    def test_infeasible_pin_is_refused_with_nothing_degraded(self):
        s = self.make_scheduler(max_ratio=0.16, overflow="lru")
        self.held_at(s, "a", 0, 1000.0)
        s.jc_pin_max_ratio = 0.05  # limit 12: a 16-token pin cannot fit alone
        b = self.start("b", self.prompt(1), OUTPUT)
        s.running_batch.reqs = s.running_batch.reqs + [b]
        out = s.pin_request(PinReqInput(rid="b"))
        self.assertFalse(out.success)
        self.assertIn("degrading every other pin would not make room", out.message)
        self.assertEqual((out.degraded, out.pins_degraded_total), ([], 0))
        self.assertEqual(set(s.pin_table), {"a"})
        self.assertEqual(b.kv.cache_protected_len, 0)

    def test_renew_repins_a_degraded_pin_like_a_new_pin(self):
        s, _ = self.three_then_a_fourth("lru")  # a degraded; b, c held
        self.now = 1003.0
        out = s.pin_request(PinReqInput(rid="a", ttl_s=60.0))
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.pinned_len, 16)
        # Admitted through the valve: b is now the least recently touched.
        self.assertEqual(out.degraded, ["b"])
        self.assertEqual((out.pins_degraded_total, out.pins_repinned_total), (2, 1))
        self.assertEqual(set(s.pin_table), {"a", "c"})
        self.assertEqual(set(s.jc_degraded_pins), {"b"})
        pin = s.pin_table["a"]
        self.assertEqual((pin.pinned_at, pin.expires_at), (1003.0, 1063.0))
        # Held again: eviction no longer reaches it.
        self.evict_all()
        self.assertEqual(self.match_len(self.prompt(0) + OUTPUT), 16)

    def test_repin_holds_what_survived_eviction(self):
        s, _ = self.three_then_a_fourth("lru")
        # Something else holds a's prompt, so eviction takes only its output.
        prompt_node = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", self.prompt(0))))
        ).last_device_node
        other = self.cache.lock(prompt_node)
        self.evict_all()
        s.unpin_request(UnpinReqInput(rid="c"))  # make room without degrading
        out = s.pin_request(PinReqInput(rid="a"))
        self.assertTrue(out.success, out.message)
        self.assertEqual((out.pinned_len, out.degraded), (len(PROMPT), []))
        self.assertEqual(s.pin_table["a"].pinned_len, len(PROMPT))
        self.cache.unlock(other)

    def test_repin_of_an_evicted_pin_fails(self):
        s, _ = self.three_then_a_fourth("lru")
        self.evict_all()
        out = s.pin_request(PinReqInput(rid="a"))
        self.assertFalse(out.success)
        self.assertIn("degraded pin's KV was evicted", out.message)
        self.assertEqual((out.pinned_len, out.pins_repinned_total), (0, 0))
        self.assertEqual(set(s.pin_table), {"b", "c"})
        out = s.pin_request(PinReqInput(rid="a"))
        self.assertEqual(out.message, "request a is not running")

    def test_repin_in_refuse_mode_is_refused_over_the_cap(self):
        s, _ = self.three_then_a_fourth("lru")
        s.jc_pin_overflow = "refuse"
        out = s.pin_request(PinReqInput(rid="a"))
        self.assertFalse(out.success)
        self.assertIn("pin cap exceeded", out.message)
        self.assertEqual(set(s.pin_table), {"b", "c"})
        self.assertIn("a", s.jc_degraded_pins)

    def test_unpin_after_degrade(self):
        s, _ = self.three_then_a_fourth("lru")
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertIn("degraded", out.message)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertEqual(out.message, "request a was not pinned")
        # The degraded mark outlives no lease: forgotten after --jc-pin-max-ttl-s.
        self.held_at(s, "e", 4, 1010.0)  # degrades b
        self.now = 1010.0 + 3600.0
        s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))
        out = s.pin_request(PinReqInput(rid="b"))
        self.assertEqual(out.message, "request b is not running")


class TestJointCheckpointPinPressureValveShared(_SharedPrefixFixture):
    """Room is counted in distinct tree tokens: degrading a pin whose path
    the kept pins or the new hold share frees nothing, so it is spared."""

    def test_only_pins_that_free_room_are_degraded(self):
        # limit 12000 (int(0.3663 x 32768)): prefix 4300 + two tails fit
        # (11700), a third tail does not (15400).
        s = self.make_scheduler(max_ratio=0.3663, overflow="lru")
        prefix_only = self.start("p", SHARED_PREFIX + [1, 2, 3], [7])
        s.running_batch.reqs = s.running_batch.reqs + [prefix_only]
        self.now = 999.0
        out = s.pin_request(PinReqInput(rid="p", up_to=PREFIX_LEN))
        self.assertEqual(out.pinned_len, PREFIX_LEN)
        self.now = 1000.0
        self.hold(s, 0)
        self.now = 1001.0
        self.hold(s, 1)
        self.assertEqual(s._jc_pinned_tokens_total(), PREFIX_LEN + 2 * TAIL_LEN)
        self.now = 1002.0
        r2 = self.hold(s, 2)
        # p is the least recently touched, but its prefix stays held by r1
        # and r2: degrading it frees nothing. r0's tail is what makes room.
        self.assertEqual(r2.jc_pin_outcome["jc_pin_degraded"], ["r0"])
        self.assertEqual(set(s.pin_table), {"p", "r1", "r2"})
        self.assertEqual(s._jc_pinned_tokens_total(), PREFIX_LEN + 2 * TAIL_LEN)



# ---------------------------------------------------------------- exclusive admission

NUM_ROWS = 16
CHUNK = 8192


class _AdmReq:
    """The fields _get_new_batch_prefill_raw and the admit-first hooks read."""

    def __init__(self, rid, n_input=6000, admit_first=False, truncate_pos=None):
        self.rid = rid
        self.origin_input_ids = array("q", range(n_input))
        self.jc_admit_first = admit_first
        self.jc_truncate_pos = truncate_pos
        self.session = None
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0
        self.next_round_inits = 0

    def init_next_round_input(self, tree_cache=None):
        self.next_round_inits += 1

    def __repr__(self):
        return self.rid


class _Adder:
    """Admits everything offered: the continuing chunk first, then waiting
    requests in order. Records the chunk size it was built with."""

    def __init__(self, *args, **kwargs):
        self.chunked_prefill_size = args[6]
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = self.chunked_prefill_size

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return req  # still chunked after this round

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestJointCheckpointAdmitFirstExclusive(CustomTestCase):
    """_get_new_batch_prefill_raw with a pass-through PrefillAdder: which
    requests reach ScheduleBatch.init_new, and whether the chunked request is
    part of the batch."""

    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.unlocked = []

    def make_scheduler(self, waiting, chunked=None, enable=True) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.jc_admit_first_max_tokens = 512
        s._jc_admit_first_streak = 0
        s.pin_table = {}
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = list(waiting)
        s.chunked_req = chunked
        s.min_free_slots_delayer = None
        s.policy = MagicMock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = CHUNK
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None,
            storage_prefetch_retries=None,
            unlock=self.unlocked.append,
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = NUM_ROWS
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: NUM_ROWS // 2)
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        return s

    def successor(self, rid, appended=16, held=4000):
        return _AdmReq(
            rid, n_input=held + appended, admit_first=True, truncate_pos=held
        )

    def admit(self, s, running_batch=None):
        """One admission round: (admitted reqs, init_new's chunked_req, adder
        chunk size), or None when nothing was admitted."""
        running_batch = running_batch or ScheduleBatch(reqs=[])
        built = {}
        real_adder = s.get_prefill_adder_class()

        def make_adder(*args, **kwargs):
            built["adder"] = real_adder(*args, **kwargs)
            return built["adder"]

        s.get_prefill_adder_class = lambda: make_adder

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list), kwargs.get("chunked_req"))

        try:
            with (
                get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
                patch.object(ScheduleBatch, "init_new", side_effect=capture),
                patch("sglang.srt.managers.scheduler.set_time_batch"),
            ):
                try:
                    s._get_new_batch_prefill_raw(
                        prefill_delayer_single_pass=None,
                        running_batch=running_batch,
                    )
                except _BatchBuilt as e:
                    reqs, chunked_kw = e.args
                    return reqs, chunked_kw, built["adder"].chunked_prefill_size
                return None
        finally:
            s.get_prefill_adder_class = lambda: real_adder

    # ------------------------------------------------------------ exclusive batch

    def test_successor_admitted_alone_among_cold_requests(self):
        cold = [_AdmReq(f"c{i}") for i in range(3)]
        succ = self.successor("s1")
        s = self.make_scheduler([cold[0], cold[1], succ, cold[2]])
        reqs, chunked_kw, chunk_size = self.admit(s)
        self.assertEqual(reqs, [succ])
        self.assertIsNone(chunked_kw)
        # No chunking in the exclusive step: a successor cannot become chunked.
        self.assertIsNone(chunk_size)
        self.assertEqual(s.waiting_queue, cold)
        for req in cold:
            self.assertEqual(req.next_round_inits, 0)

        # The next round is normal again.
        reqs, _, chunk_size = self.admit(s)
        self.assertEqual(reqs, cold)
        self.assertEqual(chunk_size, CHUNK)
        self.assertEqual(s._jc_admit_first_streak, 0)

    def test_several_successors_share_the_exclusive_batch(self):
        cold = _AdmReq("c0")
        a, b = self.successor("a", appended=100), self.successor("b", appended=200)
        s = self.make_scheduler([cold, a, b])
        reqs, _, _ = self.admit(s)
        self.assertEqual(reqs, [a, b])
        self.assertEqual(s.waiting_queue, [cold])

    def test_chunked_req_sits_out_the_exclusive_round(self):
        chunk = _AdmReq("chunk", n_input=24000)
        cold = _AdmReq("c0")
        succ = self.successor("s1")
        s = self.make_scheduler([cold, succ], chunked=chunk)
        reqs, chunked_kw, _ = self.admit(s)
        self.assertEqual(reqs, [succ])
        self.assertIsNone(chunked_kw)
        # Untouched: no next-round setup, no in-flight middle chunk counted.
        self.assertIs(s.chunked_req, chunk)
        self.assertEqual(chunk.next_round_inits, 0)
        self.assertEqual(chunk.inflight_middle_chunks, 0)

        # Next round: the chunk continues first, as without the successor.
        reqs, chunked_kw, chunk_size = self.admit(s)
        self.assertEqual(reqs, [chunk, cold])
        self.assertIs(chunked_kw, chunk)
        self.assertEqual(chunk_size, CHUNK)
        self.assertEqual(chunk.next_round_inits, 1)
        self.assertEqual(chunk.inflight_middle_chunks, 1)

    def test_full_running_batch_keeps_normal_admission(self):
        chunk = _AdmReq("chunk", n_input=24000)
        succ = self.successor("s1")
        s = self.make_scheduler([succ], chunked=chunk)
        running = ScheduleBatch(reqs=[_AdmReq("r0")])
        running.batch_is_full = True
        reqs, chunked_kw, _ = self.admit(s, running)
        # The chunk proceeds; the full batch stops waiting admission.
        self.assertEqual(reqs, [chunk])
        self.assertIs(chunked_kw, chunk)
        self.assertEqual(s.waiting_queue, [succ])

    # ------------------------------------------------------------ bounds

    def test_oversized_successor_only_keeps_head_of_queue(self):
        chunk = _AdmReq("chunk", n_input=24000)
        cold = _AdmReq("c0")
        big = self.successor("big", appended=600)
        s = self.make_scheduler([cold, big], chunked=chunk)
        reqs, chunked_kw, chunk_size = self.admit(s)
        self.assertEqual(reqs, [chunk, big, cold])
        self.assertIs(chunked_kw, chunk)
        self.assertEqual(chunk_size, CHUNK)

    def test_budget_takes_successors_from_the_head_until_full(self):
        cold = _AdmReq("c0")
        a = self.successor("a", appended=300)
        b = self.successor("b", appended=300)
        s = self.make_scheduler([cold, a, b])
        reqs, _, _ = self.admit(s)
        self.assertEqual(reqs, [a])
        # b waits for the next round, still ahead of the cold request.
        self.assertEqual(s.waiting_queue, [b, cold])

    def test_nonpositive_budget_disables_exclusive_step(self):
        cold = _AdmReq("c0")
        succ = self.successor("s1")
        s = self.make_scheduler([cold, succ])
        s.jc_admit_first_max_tokens = 0
        reqs, _, chunk_size = self.admit(s)
        self.assertEqual(reqs, [succ, cold])
        self.assertEqual(chunk_size, CHUNK)

    def test_consecutive_exclusive_steps_are_capped(self):
        chunk = _AdmReq("chunk", n_input=100000)
        cold = _AdmReq("c0")
        s = self.make_scheduler([cold], chunked=chunk)
        cap = Scheduler.JC_ADMIT_FIRST_MAX_CONSECUTIVE
        for i in range(cap):
            succ = self.successor(f"s{i}")
            s.waiting_queue.append(succ)
            reqs, chunked_kw, _ = self.admit(s)
            self.assertEqual(reqs, [succ], f"round {i}")
            self.assertIsNone(chunked_kw)
        # One more successor arrives, but a normal round is due.
        late = self.successor("late")
        s.waiting_queue.append(late)
        reqs, chunked_kw, _ = self.admit(s)
        self.assertEqual(reqs, [chunk, late, cold])
        self.assertIs(chunked_kw, chunk)
        self.assertEqual(s._jc_admit_first_streak, 0)
        # The streak starts over.
        again = self.successor("again")
        s.waiting_queue = [_AdmReq("c1"), again]
        reqs, _, _ = self.admit(s)
        self.assertEqual(reqs, [again])

    def test_not_on_disaggregated_prefill(self):
        cold = _AdmReq("c0")
        succ = self.successor("s1")
        s = self.make_scheduler([cold, succ])
        s.disaggregation_mode = DisaggregationMode.PREFILL
        reqs, _, _ = self.admit(s)
        self.assertEqual(reqs, [succ, cold])

    # ------------------------------------------------------------ estimate and holds

    def test_appended_tokens_estimate(self):
        s = self.make_scheduler([])
        self.assertEqual(s._jc_appended_tokens(self.successor("a", appended=7)), 7)
        # No appended token: the match stops one short, one token to extend.
        self.assertEqual(s._jc_appended_tokens(self.successor("b", appended=0)), 1)
        # Page alignment rounds the held prefix down.
        s.page_size = 64
        self.assertEqual(
            s._jc_appended_tokens(self.successor("c", appended=7, held=4000)),
            4007 - 3968,
        )
        # Without a truncate position the whole input counts.
        self.assertEqual(s._jc_appended_tokens(_AdmReq("d", n_input=300)), 300)

    def test_admission_drops_truncate_holds(self):
        cold = _AdmReq("c0")
        a, b = self.successor("a"), self.successor("b", appended=600)
        s = self.make_scheduler([cold, a, b])
        hold = lambda rid: JcPin(  # noqa: E731
            lock=f"lock-{rid}",
            pinned_len=4000,
            pinned_at=0.0,
            expires_at=30.0,
            successor_of="old",
        )
        s.pin_table = {"a": hold("a"), "b": hold("b")}
        reqs, _, _ = self.admit(s)
        # Only a fits the exclusive step; b keeps its hold until admitted.
        self.assertEqual(reqs, [a])
        self.assertEqual(set(s.pin_table), {"b"})
        self.assertEqual(self.unlocked, ["lock-a"])
        reqs, _, _ = self.admit(s)
        self.assertEqual(reqs, [b, cold])
        self.assertEqual(s.pin_table, {})
        self.assertEqual(self.unlocked, ["lock-a", "lock-b"])

    # ------------------------------------------------------------ flag off

    def test_flag_off_admits_like_unmarked_requests(self):
        def run(enable, marked):
            chunk = _AdmReq("chunk", n_input=24000)
            cold = [_AdmReq(f"c{i}") for i in range(2)]
            succ = self.successor("s1")
            succ.jc_admit_first = marked
            s = self.make_scheduler(
                [cold[0], succ, cold[1]], chunked=chunk, enable=enable
            )
            reqs, chunked_kw, chunk_size = self.admit(s)
            return (
                [r.rid for r in reqs],
                chunked_kw.rid,
                chunk_size,
                chunk.inflight_middle_chunks,
                [r.rid for r in s.waiting_queue],
            )

        baseline = run(enable=False, marked=False)
        self.assertEqual(baseline, (["chunk", "c0", "s1", "c1"], "chunk", CHUNK, 1, []))
        self.assertEqual(run(enable=False, marked=True), baseline)
        self.assertEqual(run(enable=True, marked=False), baseline)


if __name__ == "__main__":
    unittest.main()
