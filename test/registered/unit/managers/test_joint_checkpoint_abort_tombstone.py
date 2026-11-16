"""Joint checkpoint: an abort is never lost, whatever order it arrives in
relative to its own /generate request (single tokenizer worker).

- Window 1: the abort reaches TokenizerManager.abort_request before
  generate_request registered the rid. A tombstone (AbortTombstones) holds it;
  registration consumes it.
- Window 2: the rid is registered but the request has not been pushed to the
  scheduler yet. The AbortReq is held and sent right after the request, so the
  scheduler sees the request first.
In both cases the client gets a final chunk with finish_reason.type == "abort".
With the flag off both functions keep upstream behavior.

TokenizerManager is built with __new__; a tiny fake scheduler stands in for
the ZMQ socket: it queues requests in arrival order, aborts by rid prefix like
Scheduler.abort_request, and echoes an AbortReq back (_handle_abort_req).
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import abort_tombstones as abort_tombstones_mod  # noqa: E402
from sglang.srt.managers.abort_tombstones import AbortTombstones  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DummyAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeScheduler:
    """Receives what TokenizerManager._dispatch_to_scheduler pushes, in order."""

    def __init__(self, tm):
        self.tm = tm
        self.sent = []
        self.waiting = []  # rids of queued requests

    def __call__(self, msg):
        self.sent.append(msg)
        if isinstance(msg, AbortReq):
            hit = [r for r in self.waiting if r.startswith(msg.rid)]
            self.waiting = [r for r in self.waiting if r not in hit]
            for r in hit:
                asyncio.get_running_loop().call_soon(
                    self.tm._handle_abort_req, AbortReq(rid=r)
                )
        else:
            self.waiting.append(msg.rid)

    def aborts(self):
        return [m for m in self.sent if isinstance(m, AbortReq)]

    def requests(self):
        return [m for m in self.sent if not isinstance(m, AbortReq)]


def _make_tm(case, jc=True) -> TokenizerManager:
    override = get_context().override_server_args(
        speculative_algorithm=None,
        enable_joint_checkpoint=jc,
        tokenizer_worker_num=1,
    )
    override.install()
    case.addCleanup(override.restore)

    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm._config_updates = []
    tm.rid_to_state = {}
    tm.abort_tombstones = AbortTombstones()
    tm.encoder_dispatch_ready = {}
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.skip_tokenizer_init = False
    tm.disaggregation_mode = "none"
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.request_metrics_exporter_manager = Mock()
    tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
    tm._lora_release_tasks = set()
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    tm.cuda_vmm_feature_transport = Mock()
    tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
        return_value=[]
    )
    tm.positional_embeds_transport = Mock()
    tm.positional_embeds_transport.prepare_for_dispatch = Mock(return_value=[])
    tm.scheduler = _FakeScheduler(tm)
    tm._dispatch_to_scheduler = Mock(side_effect=tm.scheduler)

    async def _tokenize(obj):
        tok = MagicMock()
        tok.rid = obj.rid
        tok.mm_inputs = None
        return tok

    tm._tokenize_one_request = AsyncMock(side_effect=_tokenize)
    return tm


def _gen_obj(rid, stream=True):
    return GenerateReqInput(
        input_ids=[1, 2, 3],
        rid=rid,
        stream=stream,
        sampling_params={"max_new_tokens": 8},
    )


async def _collect(tm, obj):
    return [out async for out in tm.generate_request(obj)]


class TestAbortBeforeRegistration(CustomTestCase):
    """Window 1: the abort handler runs before generate_request registers."""

    def test_abort_first_then_generate_is_aborted(self):
        tm = _make_tm(self)
        tm.abort_request("r1", discard=True)
        self.assertEqual(tm.scheduler.sent, [], "nothing to send before the request")
        self.assertEqual(len(tm.abort_tombstones), 1)

        outs = asyncio.run(_collect(tm, _gen_obj("r1")))

        self.assertEqual(outs[-1]["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(outs[-1]["output_ids"], [])
        # Request first, then exactly one AbortReq carrying discard.
        self.assertEqual(
            [type(m).__name__ for m in tm.scheduler.sent][1:], ["AbortReq"]
        )
        self.assertEqual(tm.scheduler.sent[0].rid, "r1")
        (abort,) = tm.scheduler.aborts()
        self.assertEqual(abort.rid, "r1")
        self.assertTrue(abort.discard)
        self.assertEqual(tm.scheduler.waiting, [])
        self.assertNotIn("r1", tm.rid_to_state)
        self.assertEqual(len(tm.abort_tombstones), 0)

    def test_discard_false_is_carried(self):
        tm = _make_tm(self)
        tm.abort_request("r1")
        asyncio.run(_collect(tm, _gen_obj("r1", stream=False)))
        (abort,) = tm.scheduler.aborts()
        self.assertFalse(abort.discard)

    def test_repeated_abort_before_registration_sends_once(self):
        tm = _make_tm(self)
        tm.abort_request("r1")
        tm.abort_request("r1", discard=True)
        self.assertEqual(len(tm.abort_tombstones), 1)
        outs = asyncio.run(_collect(tm, _gen_obj("r1")))
        self.assertEqual(outs[-1]["meta_info"]["finish_reason"]["type"], "abort")
        (abort,) = tm.scheduler.aborts()
        self.assertTrue(abort.discard, "discard sticks if any abort asked for it")

    def test_tombstone_is_consumed_once(self):
        tm = _make_tm(self)
        tm.abort_request("r1")
        asyncio.run(_collect(tm, _gen_obj("r1")))
        # A later request reusing the rid is not affected.
        tm._init_req_state(_gen_obj("r1"))
        state = tm.rid_to_state["r1"]
        self.assertFalse(state.abort_sent)
        self.assertIsNone(state.pending_abort_discard)

    def test_exact_rid_match_only(self):
        tm = _make_tm(self)
        tm.abort_request("a-1")
        tm._init_req_state(_gen_obj("a-12"))
        state = tm.rid_to_state["a-12"]
        self.assertFalse(state.abort_sent)
        self.assertIsNone(state.pending_abort_discard)
        self.assertEqual(len(tm.abort_tombstones), 1)

    def test_batch_sub_rid_is_matched_exactly(self):
        tm = _make_tm(self)
        tm.abort_request("b1")
        obj = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            rid=["b0", "b1"],
            sampling_params={"max_new_tokens": 8},
        )
        obj.normalize_batch_and_arguments()
        tm._init_req_state(obj)
        self.assertIsNone(tm.rid_to_state["b0"].pending_abort_discard)
        self.assertIs(tm.rid_to_state["b1"].pending_abort_discard, False)
        self.assertTrue(tm.rid_to_state["b1"].abort_sent)

    def test_expired_tombstone_does_not_affect_new_request(self):
        tm = _make_tm(self)
        with patch.object(abort_tombstones_mod.time, "monotonic", return_value=100.0):
            tm.abort_request("r1")
        later = 100.0 + AbortTombstones.TTL_S + 0.1
        with patch.object(abort_tombstones_mod.time, "monotonic", return_value=later):
            tm._init_req_state(_gen_obj("r1"))
        state = tm.rid_to_state["r1"]
        self.assertFalse(state.abort_sent)
        self.assertIsNone(state.pending_abort_discard)
        self.assertEqual(len(tm.abort_tombstones), 0)


class TestAbortAfterRegistration(CustomTestCase):
    def test_normal_order_unchanged(self):
        """generate dispatched first, then abort: one AbortReq, sent at once."""
        tm = _make_tm(self)

        async def drive():
            gen = tm.generate_request(_gen_obj("r1"))
            task = asyncio.ensure_future(gen.__anext__())
            await asyncio.sleep(0)
            self.assertTrue(tm.rid_to_state["r1"].dispatched)
            tm.abort_request("r1", discard=True)
            self.assertEqual(len(tm.scheduler.aborts()), 1)
            tm.abort_request("r1", discard=True)  # deduplicated
            return await task

        out = asyncio.run(drive())
        self.assertEqual(out["meta_info"]["finish_reason"]["type"], "abort")
        (abort,) = tm.scheduler.aborts()
        self.assertTrue(abort.discard)
        self.assertEqual(len(tm.abort_tombstones), 0)

    def test_registered_not_dispatched_abort_is_sent_after_request(self):
        """Window 2: abort while generate_request waits before dispatch."""
        tm = _make_tm(self)
        tm.is_pause = True  # generate_request blocks after registration

        async def drive():
            task = asyncio.ensure_future(_collect(tm, _gen_obj("r1")))
            await asyncio.sleep(0)
            state = tm.rid_to_state["r1"]
            self.assertFalse(state.dispatched)
            tm.abort_request("r1", discard=True)
            tm.abort_request("r1")  # deduplicated
            self.assertEqual(tm.scheduler.sent, [], "abort held until dispatch")
            self.assertTrue(state.pending_abort_discard)
            async with tm.is_pause_cond:
                tm.is_pause = False
                tm.is_pause_cond.notify_all()
            return await task

        outs = asyncio.run(drive())
        self.assertEqual(outs[-1]["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(
            [type(m).__name__ for m in tm.scheduler.sent][1:], ["AbortReq"]
        )
        self.assertEqual(tm.scheduler.sent[0].rid, "r1")
        (abort,) = tm.scheduler.aborts()
        self.assertTrue(abort.discard)
        self.assertEqual(tm.scheduler.waiting, [])
        self.assertNotIn("r1", tm.rid_to_state)

    def test_undispatched_failure_drops_pending_abort(self):
        """A request that fails before dispatch never sends its held abort."""
        tm = _make_tm(self)
        tm.abort_request("r1")
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("too long"))
        with self.assertRaises(ValueError):
            asyncio.run(_collect(tm, _gen_obj("r1")))
        self.assertEqual(tm.scheduler.sent, [])
        self.assertNotIn("r1", tm.rid_to_state)


class TestAbortTombstones(CustomTestCase):
    def test_capacity_evicts_oldest(self):
        t = AbortTombstones(max_entries=3)
        for rid in ["a", "b", "c", "d"]:
            t.add(rid, discard=False)
        self.assertEqual(len(t), 3)
        self.assertIsNone(t.pop("a"))
        self.assertIs(t.pop("b"), False)
        self.assertIs(t.pop("d"), False)

    def test_expiry_prunes_on_access(self):
        t = AbortTombstones(ttl_s=10.0)
        with patch.object(abort_tombstones_mod.time, "monotonic", return_value=0.0):
            t.add("a", discard=True)
        with patch.object(abort_tombstones_mod.time, "monotonic", return_value=5.0):
            t.add("b", discard=False)
        with patch.object(abort_tombstones_mod.time, "monotonic", return_value=12.0):
            self.assertIsNone(t.pop("a"))
            self.assertEqual(len(t), 1)
            self.assertIs(t.pop("b"), False)

    def test_tokenizer_manager_capacity_bound(self):
        tm = _make_tm(self)
        tm.abort_tombstones = AbortTombstones(max_entries=2)
        for rid in ["x0", "x1", "x2"]:
            tm.abort_request(rid)
        self.assertEqual(len(tm.abort_tombstones), 2)
        tm._init_req_state(_gen_obj("x0"))
        self.assertFalse(tm.rid_to_state["x0"].abort_sent)


class TestFlagOff(CustomTestCase):
    def test_unregistered_abort_returns_without_tombstone(self):
        tm = _make_tm(self, jc=False)
        tm.abort_request("r1", discard=True)
        self.assertEqual(tm.scheduler.sent, [])
        self.assertEqual(len(tm.abort_tombstones), 0)
        tm._init_req_state(_gen_obj("r1"))
        self.assertFalse(tm.rid_to_state["r1"].abort_sent)

    def test_registered_not_dispatched_abort_is_sent_immediately(self):
        tm = _make_tm(self, jc=False)
        tm._init_req_state(_gen_obj("r1"))
        tm.abort_request("r1", discard=True)
        (abort,) = tm.scheduler.sent
        self.assertIsInstance(abort, AbortReq)
        self.assertTrue(abort.discard)
        state = tm.rid_to_state["r1"]
        self.assertTrue(state.abort_sent)
        self.assertIsNone(state.pending_abort_discard)
        # Upstream: a later dispatch does not re-send it.
        tm._mark_state_dispatched("r1")
        self.assertEqual(len(tm.scheduler.sent), 1)


if __name__ == "__main__":
    unittest.main()
