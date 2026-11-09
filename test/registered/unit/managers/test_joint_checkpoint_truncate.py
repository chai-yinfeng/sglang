"""Joint checkpoint P3 (truncate, path A') on a CPU-only Scheduler built with
Scheduler.__new__, a real streaming Session, a real StreamingSession slot
over fake pools, and real Req objects.

- Scheduler._jc_truncate_session finishes the session's in-flight request at
  session_params.truncate_to without a forward: output_ids are cut, the
  final chunk is streamed, and release_kv_cache keeps the record in the slot
  (kv_committed_len == pos, tail freed, tree lock still the slot's).
- Session.create_req then builds the successor as a plain streaming append
  (prompt[:pos] + append_ids); its slot match gives prefix_len == pos, so only
  append_ids need prefill budget, and _jc_front_waiting_queue admits it ahead
  of the waiting queue in the same step.
- Rejections: flag off, non-streaming session, no in-flight request, pos out
  of range, pos below cache_protected_len, request already finishing.
- Flag off: a streaming turn aborted the upstream way still hands the record
  back and frees it (StreamingSession.try_cache_finished_req unchanged).
- Overlap: a late result for the truncated request streams nothing more and
  frees nothing more.
"""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import SessionParams
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import StreamingSession
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROMPT = [11, 12, 13, 14]
VOCAB = 1 << 20


class _FakeAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double that records what is freed."""

    def __init__(self, page_size: int = 1):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())

    def freed_positions(self):
        return sorted(int(x) for t in self.freed for x in t.tolist())


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.free_slots = []

    def free(self, req):
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


class _FakeCache:
    """Stands in for UnifiedRadixCache: owns a real StreamingSession and
    tries it first; the tree itself never matches."""

    free_kv_row = BasePrefixCache.free_kv_row

    def __init__(self, page_size: int = 1):
        # Row i holds positions [i*128, (i+1)*128) so freed indices name positions.
        self.req_to_token_pool = _FakeReqToTokenPool(
            torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128)
        )
        self.token_to_kv_pool_allocator = _FakeAllocator(page_size)
        self.page_size = page_size
        self.session = StreamingSession(self)
        self.unlocked = []
        self.checkpoints = []
        self.tree_matches = 0

    def supports_mamba(self):
        return False

    def swa_reprefill_tail_tokens(self, req):
        return 0

    def checkpoint(self, req, *, up_to, **kwargs):
        self.checkpoints.append((req.rid, up_to))

    def maybe_hand_to_session(self, req):
        self.session.take(req)

    def claim_kv_row(self, req):
        return self.session.try_cache_finished_req(req)

    def on_release(self, req, *, checkpointed):
        pass

    def match_prefix(self, params):
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        self.tree_matches += 1
        return MatchResult(
            device_prefix_len=0,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def unlock(self, lock):
        self.unlocked.append(lock)


def _recv(rid, input_ids, truncate_to=None, session_rid=None, max_new_tokens=100):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        input_text=None,
        mm_inputs=None,
        session_params=SessionParams(id="s", rid=session_rid, truncate_to=truncate_to),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=True,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        sampling_logprobs_mode="selected",
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        priority=None,
        routing_key=None,
        extra_key=None,
        cache_salt=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestJointCheckpointTruncate(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.cache = _FakeCache()
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    # ------------------------------------------------------------ fixtures

    def start_turn(self, rid="t1", output_len=5, protected_len=0) -> Req:
        """A streaming turn that was admitted, prefilled and decoded
        output_len tokens; under overlap one more decode slot is allocated."""
        req = self.session.create_req(
            _recv(rid, PROMPT), tokenizer=None, vocab_size=VOCAB
        )
        self.assertFalse(req.finished())
        req.output_ids.extend(range(100, 100 + output_len))
        # prepare_for_decode keeps the fill array at origin + output_ids.
        req.refresh_fill_ids()
        total = len(PROMPT) + output_len
        req.kv.req_pool_idx = 0
        req.kv.cache_protected_len = protected_len
        req.kv.kv_committed_len = total + 1
        req.kv.kv_allocated_len = total + 1
        req.lock = TreeLock(node="admission-node", receipt=None)
        # First row allocation: the session takes the record and the lock.
        self.cache.maybe_hand_to_session(req)
        return req

    def finish_turn(self, req: Req) -> None:
        """The turn ends naturally (length); its record stays in the slot."""
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        req.finished_len = len(req.output_ids)
        release_kv_cache(req, self.cache, checkpoint=True)
        self.assertFalse(self.session._inflight)
        self.cache.token_to_kv_pool_allocator.clear()

    def make_batch(self, reqs) -> ScheduleBatch:
        batch = ScheduleBatch(reqs=list(reqs), batch_is_full=False)
        batch.device = "cpu"
        batch.req_to_token_pool = self.cache.req_to_token_pool
        batch.token_to_kv_pool_allocator = self.cache.token_to_kv_pool_allocator
        batch.tree_cache = self.cache
        batch.model_config = MagicMock(is_encoder_decoder=False)
        batch.sampling_info = MagicMock()
        batch.spec_info = None
        rows = [r.kv.req_pool_idx for r in reqs]
        lens = [len(r.origin_input_ids) + len(r.output_ids) for r in reqs]
        batch.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.seq_lens = torch.tensor(lens, dtype=torch.int64)
        batch.seq_lens_cpu = torch.tensor(lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(lens, dtype=torch.int32)
        batch.multimodal_inputs = [None for _ in reqs]
        batch.input_ids = None
        return batch

    def make_scheduler(self, running, enable=True, streamer=None) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.paused_reqs = {}
        s.chunked_req = None
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.tree_cache = self.cache
        s.req_to_token_pool = self.cache.req_to_token_pool
        s.token_to_kv_pool_allocator = self.cache.token_to_kv_pool_allocator
        s.running_batch = self.make_batch(running)
        s.last_batch = None
        s.result_queue = deque()
        s.enable_continuous_input_polling = False
        s.waiting_queue = []
        s.draft_worker = None
        s.disaggregation_mode = DisaggregationMode.NULL
        s.output_streamer = streamer if streamer is not None else MagicMock()
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

    def truncate(self, s, pos, rid="t2", append=(7,), session_rid=None):
        recv = _recv(rid, append, truncate_to=pos, session_rid=session_rid)
        return recv, s._jc_truncate_session(self.session, recv)

    # ------------------------------------------------------------ rejections

    def test_rejected_without_flag(self):
        old = self.start_turn()
        s = self.make_scheduler([old], enable=False)
        _, err = self.truncate(s, len(PROMPT) + 2)
        self.assertIn("--enable-joint-checkpoint", err)
        self.assertFalse(old.finished())
        self.assertEqual(len(old.output_ids), 5)
        self.assertTrue(self.session._inflight)
        s.output_streamer.stream_output.assert_not_called()

    def test_rejected_for_non_streaming_session(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=False)
        s = self.make_scheduler([])
        _, err = self.truncate(s, 4)
        self.assertIn("streaming", err)

    def test_rejected_on_empty_session(self):
        s = self.make_scheduler([])
        _, err = self.truncate(s, len(PROMPT))
        self.assertIn("nothing to truncate", err)

    def test_rejected_out_of_range_and_wrong_rid(self):
        old = self.start_turn(output_len=5)
        s = self.make_scheduler([old])
        total = len(PROMPT) + 5
        for pos in (len(PROMPT) - 1, total + 1):
            _, err = self.truncate(s, pos)
            self.assertIn("outside", err, pos)
        _, err = self.truncate(s, len(PROMPT) + 1, session_rid="someone-else")
        self.assertIn("not the in-flight request", err)
        self.assertFalse(old.finished())
        # Naming the right rid is accepted.
        _, err = self.truncate(s, len(PROMPT) + 1, session_rid="t1")
        self.assertIsNone(err)

    def test_rejected_below_cache_protected_len(self):
        # Cannot happen while pos >= len(input_ids) >= the published prompt;
        # guarded anyway because the slot's lock cannot move down the tree.
        old = self.start_turn(output_len=5, protected_len=len(PROMPT) + 3)
        s = self.make_scheduler([old])
        _, err = self.truncate(s, len(PROMPT) + 2)
        self.assertIn("tree-protected prefix", err)
        self.assertFalse(old.finished())

    def test_rejected_when_finishing_or_prefilling(self):
        old = self.start_turn()
        s = self.make_scheduler([old])
        old.to_finish = FINISH_ABORT()
        _, err = self.truncate(s, len(PROMPT) + 1)
        self.assertIn("finishing", err)
        old.to_finish = None
        s.chunked_req = old
        _, err = self.truncate(s, len(PROMPT) + 1)
        self.assertIn("prefill", err)

    # ------------------------------------------------------------ the cut

    def test_truncate_finishes_old_request_and_keeps_slot(self):
        old = self.start_turn(output_len=5)
        record = old.kv
        slot = self.cache.session.slots["s"]
        slot_lock = slot.lock
        self.assertEqual(slot_lock.node, "admission-node")
        s = self.make_scheduler([old])
        pos = len(PROMPT) + 2

        recv, err = self.truncate(s, pos)
        self.assertIsNone(err)

        # Finished without a forward, cut at pos, final chunk streamed once.
        self.assertTrue(old.finished())
        self.assertIsInstance(old.finished_reason, FINISH_ABORT)
        self.assertTrue(old.finished_reason.message.startswith("Truncated"))
        self.assertEqual(old.jc_truncate_to, pos)
        self.assertEqual(old.finished_len, 2)
        self.assertEqual(list(old.output_ids), [100, 101])
        s.output_streamer.stream_output.assert_called_once()
        self.assertEqual(s.output_streamer.stream_output.call_args.args[0], [old])

        # The record stayed in the slot: committed/allocated at pos, the tail
        # (including the in-flight decode slot) freed, the lock untouched.
        self.assertIn("s", self.cache.session.slots)
        self.assertIs(slot.kv, record)
        self.assertEqual(record.kv_committed_len, pos)
        self.assertEqual(record.kv_allocated_len, pos)
        self.assertEqual(record.req_pool_idx, 0)
        self.assertIs(slot.lock, slot_lock)
        self.assertEqual(self.cache.unlocked, [])
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [])
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.freed_positions(),
            list(range(pos, len(PROMPT) + 5 + 1)),
        )
        # Nothing was written to the tree for it.
        self.assertEqual(self.cache.checkpoints, [])
        # The request itself no longer holds KV.
        self.assertFalse(old.kv.holds_kv)

        # The session is ready for the successor: the cut turn is its history.
        self.assertFalse(self.session._inflight)
        self.assertEqual(list(self.session.req_nodes), ["t1"])
        self.assertEqual(self.session.committed_origin_len, len(PROMPT))
        # The fill array is cut with it, so the share path carries it over
        # instead of rebuilding.
        self.assertEqual(list(old.full_untruncated_fill_ids), PROMPT + [100, 101])
        self.assertEqual(self.session.committed_fill_len, pos)

        # The next step's filter drops it from the running batch.
        s.running_batch.filter_batch()
        self.assertEqual(s.running_batch.reqs, [])

    def test_truncate_at_input_len_keeps_prompt_only(self):
        old = self.start_turn(output_len=3)
        s = self.make_scheduler([old])
        pos = len(PROMPT)
        _, err = self.truncate(s, pos)
        self.assertIsNone(err)
        self.assertEqual(list(old.output_ids), [])
        self.assertEqual(self.cache.session.slots["s"].kv.kv_committed_len, pos)

    def test_truncate_paused_request(self):
        old = self.start_turn(output_len=5)
        s = self.make_scheduler([])
        old.jc_pause_reason = "manual"
        s.paused_reqs[old.rid] = old
        _, err = self.truncate(s, len(PROMPT) + 1)
        self.assertIsNone(err)
        self.assertTrue(old.finished())
        # The pause bookkeeping drops it as finished at the next step.
        s.running_batch = s._jc_pause_resume(s.running_batch)
        self.assertEqual(s.paused_reqs, {})

    # ------------------------------------------------------------ the successor

    def successor_after_cut(self, pos, append=(7, 8)):
        old = self.start_turn(output_len=5)
        s = self.make_scheduler([old])
        recv, err = self.truncate(s, pos, append=append)
        self.assertIsNone(err)
        new = self.session.create_req(recv, tokenizer=None, vocab_size=VOCAB)
        new.jc_admit_first = True  # as _handle_generate_request does
        return s, old, new

    def test_successor_is_prompt_to_pos_plus_append(self):
        pos = len(PROMPT) + 2
        s, old, new = self.successor_after_cut(pos, append=(7, 8))
        self.assertFalse(new.finished(), getattr(new.to_finish, "message", None))
        self.assertEqual(list(new.origin_input_ids), PROMPT + [100, 101, 7, 8])
        # Carried fill array, no O(context) rebuild.
        self.assertEqual(list(new.full_untruncated_fill_ids), PROMPT + [100, 101, 7, 8])
        self.assertIs(new.session, self.session)
        self.assertTrue(self.session._inflight)

    def test_successor_borrows_slot_with_prefix_at_pos(self):
        pos = len(PROMPT) + 2
        s, old, new = self.successor_after_cut(pos, append=(7, 8))
        slot = self.cache.session.slots["s"]

        # What get_new_batch_prefill does per waiting request.
        new.init_next_round_input(self.cache)

        self.assertEqual(self.cache.tree_matches, 0)
        self.assertIs(new.kv, slot.kv)
        self.assertEqual(new.prefix_len, pos)
        self.assertEqual(new.kv.kv_committed_len, pos)
        self.assertIs(new.last_node, slot.virtual_node)
        # Only the appended tokens are new work for the PrefillAdder.
        self.assertEqual(len(new.full_untruncated_fill_ids) - new.prefix_len, 2)

    def test_successor_with_one_appended_token(self):
        # The match key is one token short (logit reserve); the prefix is still pos.
        pos = len(PROMPT) + 3
        s, old, new = self.successor_after_cut(pos, append=(7,))
        new.init_next_round_input(self.cache)
        self.assertEqual(new.prefix_len, pos)
        self.assertEqual(len(new.full_untruncated_fill_ids) - new.prefix_len, 1)

    def test_successor_goes_to_head_of_waiting_queue(self):
        pos = len(PROMPT) + 2
        s, old, new = self.successor_after_cut(pos)
        others = [
            Req(
                rid=f"w{i}",
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(max_new_tokens=10),
            )
            for i in range(3)
        ]
        # The policy sort may have put it anywhere; the hook fronts it.
        s.waiting_queue = [others[0], others[1], new, others[2]]
        s._jc_front_waiting_queue()
        self.assertEqual(s.waiting_queue, [new, others[0], others[1], others[2]])
        # Idempotent and order-preserving for everyone else.
        s._jc_front_waiting_queue()
        self.assertEqual(s.waiting_queue, [new, others[0], others[1], others[2]])

    # ------------------------------------------------------------ finished turn

    def test_truncate_finished_turn_cuts_slot_and_ledger(self):
        old = self.start_turn(output_len=5)
        self.finish_turn(old)
        slot = self.cache.session.slots["s"]
        slot_lock = slot.lock
        self.assertEqual(slot.kv.kv_committed_len, len(PROMPT) + 5)
        s = self.make_scheduler([])
        pos = len(PROMPT) + 2

        recv, err = self.truncate(s, pos, append=(7, 8))
        self.assertIsNone(err)

        # Nothing streamed, finish reason untouched, ledger cut.
        s.output_streamer.stream_output.assert_not_called()
        self.assertIsInstance(old.finished_reason, FINISH_LENGTH)
        self.assertEqual(list(old.output_ids), [100, 101])
        self.assertEqual(old.finished_len, 2)
        self.assertEqual(list(old.full_untruncated_fill_ids), PROMPT + [100, 101])
        self.assertEqual(self.session.committed_fill_len, pos)
        # Slot cut at pos, tail freed, lock untouched.
        self.assertEqual(slot.kv.kv_committed_len, pos)
        self.assertEqual(slot.kv.kv_allocated_len, pos)
        self.assertIs(slot.lock, slot_lock)
        self.assertEqual(self.cache.unlocked, [])
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.freed_positions(),
            list(range(pos, len(PROMPT) + 5)),
        )

        # Successor: plain append on the cut history, prefix at pos.
        new = self.session.create_req(recv, tokenizer=None, vocab_size=VOCAB)
        self.assertFalse(new.finished())
        self.assertEqual(list(new.origin_input_ids), PROMPT + [100, 101, 7, 8])
        self.assertEqual(list(new.full_untruncated_fill_ids), PROMPT + [100, 101, 7, 8])
        new.init_next_round_input(self.cache)
        self.assertIs(new.kv, slot.kv)
        self.assertEqual(new.prefix_len, pos)
        self.assertEqual(len(new.full_untruncated_fill_ids) - new.prefix_len, 2)

    def test_truncate_finished_turn_twice(self):
        old = self.start_turn(output_len=5)
        self.finish_turn(old)
        slot = self.cache.session.slots["s"]
        s = self.make_scheduler([])
        pos1, pos2 = len(PROMPT) + 4, len(PROMPT) + 1

        _, err = self.truncate(s, pos1)
        self.assertIsNone(err)
        self.assertEqual(slot.kv.kv_allocated_len, pos1)
        _, err = self.truncate(s, pos2)
        self.assertIsNone(err)

        self.assertEqual(slot.kv.kv_committed_len, pos2)
        self.assertEqual(slot.kv.kv_allocated_len, pos2)
        self.assertEqual(list(old.output_ids), [100])
        self.assertEqual(self.session.committed_fill_len, pos2)
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.freed_positions(),
            list(range(pos2, len(PROMPT) + 5)),
        )
        recv = _recv("t2", (7,), truncate_to=pos2)
        new = self.session.create_req(recv, tokenizer=None, vocab_size=VOCAB)
        self.assertEqual(list(new.origin_input_ids), PROMPT + [100, 7])
        new.init_next_round_input(self.cache)
        self.assertEqual(new.prefix_len, pos2)

    def test_truncate_finished_turn_rejections(self):
        s = self.make_scheduler([])
        _, err = self.truncate(s, len(PROMPT))
        self.assertIn("nothing to truncate", err)

        old = self.start_turn(output_len=5)
        self.finish_turn(old)
        slot = self.cache.session.slots["s"]
        total = len(PROMPT) + 5
        for pos in (len(PROMPT) - 1, total + 1):
            _, err = self.truncate(s, pos)
            self.assertIn("outside", err, pos)
        _, err = self.truncate(s, len(PROMPT) + 1, session_rid="someone-else")
        self.assertIn("not the last turn", err)
        slot.kv.cache_protected_len = len(PROMPT) + 3
        _, err = self.truncate(s, len(PROMPT) + 2)
        self.assertIn("tree-protected prefix", err)
        slot.kv.cache_protected_len = 0
        # Nothing was cut by the rejected calls.
        self.assertEqual(slot.kv.kv_committed_len, total)
        self.assertEqual(list(old.output_ids), list(range(100, 105)))
        self.assertEqual(self.cache.token_to_kv_pool_allocator.freed_positions(), [])

        self.cache.session.release_session("s")
        _, err = self.truncate(s, len(PROMPT) + 1)
        self.assertIn("holds no KV", err)

    def test_finished_turn_truncate_rejected_without_flag(self):
        old = self.start_turn(output_len=5)
        self.finish_turn(old)
        s = self.make_scheduler([], enable=False)
        _, err = self.truncate(s, len(PROMPT) + 1)
        self.assertIn("--enable-joint-checkpoint", err)
        self.assertEqual(list(old.output_ids), list(range(100, 105)))

    # ------------------------------------------------------------ flag off: upstream abort path

    def test_aborted_turn_without_truncate_releases_slot_as_upstream(self):
        old = self.start_turn(output_len=5)
        slot = self.cache.session.slots["s"]
        # The running-request abort path: one more decode step finishes it.
        old.finished_reason = FINISH_ABORT()
        old.finished_len = 5

        release_kv_cache(old, self.cache, checkpoint=True)

        self.assertNotIn("s", self.cache.session.slots)
        self.assertIsNone(slot.lock)
        self.assertEqual(
            [lock.node for lock in self.cache.unlocked], ["admission-node"]
        )
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [0])
        self.assertFalse(old.kv.holds_kv)
        self.assertFalse(self.session._inflight)
        self.assertEqual(self.session.req_nodes, {})

    # ------------------------------------------------------------ overlap

    def test_late_result_for_truncated_request_streams_and_frees_nothing(self):
        old = self.start_turn(output_len=5)
        streamer = self.real_streamer()
        s = self.make_scheduler([old], streamer=streamer)
        pos = len(PROMPT) + 2
        _, err = self.truncate(s, pos)
        self.assertIsNone(err)

        # The cut streamed exactly one payload: the final chunk, abort/truncated.
        send = streamer.send_to_detokenizer.send_output
        self.assertEqual(send.call_count, 1)
        payload = send.call_args.args[0]
        self.assertEqual(payload.rids, ["t1"])
        self.assertEqual(payload.finished_reasons[0]["type"], "abort")
        self.assertTrue(payload.finished_reasons[0]["message"].startswith("Truncated"))
        self.assertTrue(old.finished_output)

        slot = self.cache.session.slots["s"]
        freed_before = self.cache.token_to_kv_pool_allocator.freed_positions()

        # The in-flight decode result lands: process_batch_result_decode skips
        # a finished request (no output append, no release) and streams the
        # batch; the streamer skips a request whose final chunk went out.
        self.assertTrue(old.finished())
        streamer.stream_output([old], False)

        self.assertEqual(send.call_count, 1)
        self.assertEqual(list(old.output_ids), [100, 101])
        self.assertIs(slot.kv.req_pool_idx, 0)
        self.assertEqual(slot.kv.kv_committed_len, pos)
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.freed_positions(), freed_before
        )
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [])


if __name__ == "__main__":
    unittest.main()
