"""Joint checkpoint pin backstops on a CPU-only Scheduler built with
Scheduler.__new__ over a real UnifiedRadixCache:

- a pin expires after --jc-pin-ttl-s (or its own ttl_s) at the next
  scheduling step (Scheduler._jc_on_step), even under speculative decoding
  where pause/resume is skipped; re-pinning the same rid restarts the TTL;
  a TTL <= 0 or past --jc-pin-max-ttl-s is refused;
- pins are refused once the distinct tokens they hold would exceed
  --jc-pin-max-ratio x max_total_num_tokens, judged after replacing the
  rid's own earlier hold;
- flush_cache (and the PD role switch reset) drop the pin table with the tree;
- unpin is idempotent: an expired / flushed / never-pinned rid answers
  success, "was not pinned", without touching the tree; flag off still fails.
The clock is the scheduler module's ``time`` replaced by a settable stub.
"""

from sglang.srt.arg_groups.validation_hook import check_joint_checkpoint_server_args
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.managers.io_struct import PinReqInput, UnpinReqInput
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

POOL_SIZE = 64


def make_cache() -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=POOL_SIZE,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
    )


def match_len(cache, token_ids) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).device_prefix_len


class MockReq:
    """A running request that matched nothing; the fields pin_request,
    checkpoint and release_kv_cache read."""

    session = None
    session_id = None
    extra_key = None
    cache_salt = None
    priority = 0
    kv_rotation_base = None
    finished_reason = None
    skip_radix_cache_insert = False
    finished_len = None
    jc_finish_hold = None

    def __init__(self, cache, rid, fill_ids, prompt_len):
        self.rid = rid
        allocator = cache.token_to_kv_pool_allocator
        req_pool = cache.req_to_token_pool
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.origin_input_ids = array("q", fill_ids[:prompt_len])
        self.output_ids = array("q", fill_ids[prompt_len:])
        self.extend_end = len(fill_ids)
        self.prefix_len = 0
        self.last_node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", fill_ids)))
        ).last_device_node
        self.lock = cache.lock(self.last_node)
        indices = allocator.alloc(len(fill_ids))
        req_pool_idx = req_pool.alloc_rows(1)[0]
        req_pool.write((req_pool_idx, slice(0, len(fill_ids))), indices)
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=len(fill_ids),
            kv_allocated_len=len(fill_ids),
            cache_protected_len=0,
        )

    def finished(self):
        return self.finished_reason is not None

    def refresh_fill_ids(self):
        pass

    def owned_kv_len(self):
        return self.kv.kv_committed_len


class TestJointCheckpointPinTTL(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.cache = make_cache()
        self.now = 1000.0
        clock = SimpleNamespace(monotonic=lambda: self.now)
        patcher = patch.object(scheduler_module, "time", clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reqs = {}

    # ------------------------------------------------------------ fixtures

    def make_scheduler(
        self, enable=True, ttl_s=600.0, max_ratio=0.5, overflow="refuse"
    ) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = enable
        s.pin_table = {}
        s.jc_pin_ttl_s = ttl_s
        s.jc_pin_max_ttl_s = 3600.0
        s.jc_pin_max_ratio = max_ratio
        s.jc_pin_overflow = overflow
        s.jc_degraded_pins = {}
        s.jc_pins_degraded_total = 0
        s.max_total_num_tokens = POOL_SIZE
        s.tree_cache = self.cache
        s.paused_reqs = {}
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s._find_inflight_req = lambda rid: self.reqs.get(rid)
        return s

    def add_req(self, rid, base, n=20, prompt_len=10) -> MockReq:
        fill = list(range(base, base + n))
        req = MockReq(self.cache, rid, fill, prompt_len=prompt_len)
        self.reqs[rid] = req
        return req

    def finish(self, req):
        """The request goes away; only a pin keeps its prefix locked."""
        req.finished_reason = "abort"
        release_kv_cache(req, self.cache, checkpoint=True)
        del self.reqs[req.rid]

    def pin(self, s, rid, up_to, ttl_s=None):
        return s.pin_request(PinReqInput(rid=rid, up_to=up_to, ttl_s=ttl_s))

    def step(self, s):
        return s._jc_on_step(ScheduleBatch(reqs=[], batch_is_full=False))

    def evict_all(self):
        self.cache.evict(EvictParams(num_tokens=POOL_SIZE))

    # ------------------------------------------------------------ TTL

    def test_pin_reports_ttl_and_total(self):
        s = self.make_scheduler()
        self.add_req("a", 100)
        self.add_req("b", 200)
        out = self.pin(s, "a", 8)
        self.assertTrue(out.success)
        self.assertEqual(out.pinned_len, 8)
        self.assertEqual(out.expires_in_s, 600.0)
        self.assertEqual(out.pinned_tokens_total, 8)
        out = self.pin(s, "b", 5)
        self.assertEqual(out.pinned_tokens_total, 13)
        self.assertEqual(s.pin_table["a"].expires_at, 1600.0)

    def test_expired_pin_is_released_at_next_step(self):
        s = self.make_scheduler()
        req = self.add_req("a", 100)
        fill = list(req.full_untruncated_fill_ids)
        self.assertTrue(self.pin(s, "a", 12).success)
        self.finish(req)

        self.now = 1599.0
        self.step(s)
        self.evict_all()
        self.assertIn("a", s.pin_table)
        self.assertEqual(match_len(self.cache, fill), 12)

        self.now = 1600.0
        with self.assertLogs(scheduler_module.logger, level="INFO") as logs:
            self.step(s)
        self.assertNotIn("a", s.pin_table)
        self.assertTrue(
            any("pin expired rid=a len=12 after 600.0s" in m for m in logs.output)
        )
        self.evict_all()
        self.assertEqual(match_len(self.cache, fill), 0)
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.available_size(), POOL_SIZE
        )

    def test_repin_restarts_ttl(self):
        s = self.make_scheduler()
        req = self.add_req("a", 100)
        fill = list(req.full_untruncated_fill_ids)
        self.pin(s, "a", 8)
        self.now = 1500.0
        out = self.pin(s, "a", 10)
        self.assertTrue(out.success)
        self.assertEqual(out.pinned_tokens_total, 10)  # replaced, not added
        self.finish(req)

        self.now = 1700.0  # past the first pin's deadline
        self.step(s)
        self.evict_all()
        self.assertEqual(match_len(self.cache, fill), 10)

        self.now = 2100.0
        self.step(s)
        self.assertEqual(s.pin_table, {})
        self.evict_all()
        # The replaced lock was released at re-pin, the new one at expiry:
        # nothing stays locked.
        self.assertEqual(match_len(self.cache, fill), 0)

    def test_request_ttl_overrides_default(self):
        s = self.make_scheduler()
        self.add_req("short", 100)
        self.add_req("long", 200)
        out = self.pin(s, "short", 5, ttl_s=5.0)
        self.assertEqual(out.expires_in_s, 5.0)
        out = self.pin(s, "long", 5, ttl_s=3600.0)
        self.assertEqual(out.expires_in_s, 3600.0)

        self.now = 1005.0
        self.step(s)
        self.assertEqual(set(s.pin_table), {"long"})
        self.now = 4600.0
        self.step(s)
        self.assertEqual(s.pin_table, {})

    def test_ttl_must_be_positive_and_bounded(self):
        """No hold lives forever: ttl_s <= 0 or past --jc-pin-max-ttl-s is
        refused without touching the tree, and so is such a server default."""
        s = self.make_scheduler()
        self.add_req("a", 100)
        for ttl_s, needle in (
            (0, "ttl_s must be > 0, got 0"),
            (-1.0, "ttl_s must be > 0, got -1.0"),
            (3600.5, "ttl_s 3600.5 exceeds --jc-pin-max-ttl-s 3600.0"),
        ):
            out = self.pin(s, "a", 5, ttl_s=ttl_s)
            self.assertFalse(out.success)
            self.assertEqual(out.message, needle)
        self.assertEqual(s.pin_table, {})
        self.assertEqual(self.reqs["a"].kv.cache_protected_len, 0)
        for ttl, max_ttl in ((0.0, 3600.0), (601.0, 600.0), (600.0, 0.0)):
            args = ServerArgs(
                model_path="dummy",
                enable_joint_checkpoint=True,
                jc_pin_ttl_s=ttl,
                jc_pin_max_ttl_s=max_ttl,
            )
            with self.assertRaises(ValueError, msg=(ttl, max_ttl)):
                check_joint_checkpoint_server_args(args)
        check_joint_checkpoint_server_args(ServerArgs(model_path="dummy"))

    def test_expiry_runs_under_speculative_decoding(self):
        s = self.make_scheduler()
        s.spec_algorithm = SpeculativeAlgorithm.EAGLE
        self.add_req("a", 100)
        self.pin(s, "a", 5)
        batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.now = 1600.0
        self.assertIs(s._jc_on_step(batch), batch)
        self.assertEqual(s.pin_table, {})

    def test_unpin_is_idempotent(self):
        s = self.make_scheduler()
        self.add_req("a", 100)
        self.pin(s, "a", 5)
        s.tree_cache = MagicMock(wraps=self.cache)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertEqual(out.message, "")
        s.tree_cache.unlock.assert_called_once()
        # Unpinning again, or a rid never pinned: success, nothing unlocked.
        for rid in ("a", "never"):
            out = s.unpin_request(UnpinReqInput(rid=rid))
            self.assertTrue(out.success)
            self.assertEqual(out.message, f"request {rid} was not pinned")
        s.tree_cache.unlock.assert_called_once()

    def test_unpin_expired_succeeds_as_not_pinned(self):
        s = self.make_scheduler()
        self.add_req("a", 100)
        self.pin(s, "a", 5)
        self.now = 2000.0
        self.step(s)
        s.tree_cache = MagicMock(wraps=self.cache)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertEqual(out.message, "request a was not pinned")
        s.tree_cache.unlock.assert_not_called()

    # ------------------------------------------------------------ cap

    def test_cap_refuses_new_pin_and_judges_repin_after_replacement(self):
        s = self.make_scheduler(max_ratio=0.5)  # limit 32 of 64
        self.add_req("a", 100, n=30)
        self.add_req("b", 200, n=30)
        self.assertTrue(self.pin(s, "a", 20).success)
        out = self.pin(s, "b", 13)
        self.assertFalse(out.success)
        self.assertIn("pin cap exceeded", out.message)
        self.assertIn(
            "33 distinct pinned tokens with this hold on b (20 without it) > limit 32",
            out.message,
        )
        self.assertEqual(out.pinned_tokens_total, 20)
        self.assertNotIn("b", s.pin_table)
        # The refused pin touched nothing: b's KV is still its own.
        self.assertEqual(self.reqs["b"].kv.cache_protected_len, 0)

        self.assertTrue(self.pin(s, "b", 10).success)  # 30 <= 32
        # Re-pinning a counts a's old hold out: 10 + 22 = 32, not 10 + 20 + 22.
        out = self.pin(s, "a", 22)
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.pinned_tokens_total, 32)
        out = self.pin(s, "a", 23)
        self.assertFalse(out.success)
        # The refused re-pin keeps the earlier hold.
        self.assertEqual(s.pin_table["a"].pinned_len, 22)

    def test_cap_disabled_with_nonpositive_ratio(self):
        s = self.make_scheduler(max_ratio=0)
        self.add_req("a", 100, n=30)
        self.add_req("b", 200, n=30)
        self.assertTrue(self.pin(s, "a", 30).success)
        out = self.pin(s, "b", 30)
        self.assertTrue(out.success)
        self.assertEqual(out.pinned_tokens_total, 60)

    # ------------------------------------------------------------ cleanup paths

    def flush_ready(self, s):
        s.is_fully_idle = lambda **kwargs: True
        s._engine_paused = False
        s.tree_cache = MagicMock(wraps=self.cache)
        s.req_to_token_pool = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        s.grammar_manager = MagicMock()
        s.metrics_reporter = MagicMock()
        s.draft_worker = None

    def test_flush_with_pins_succeeds_and_clears_table(self):
        s = self.make_scheduler()
        self.add_req("a", 100)
        self.pin(s, "a", 5)
        self.flush_ready(s)
        self.assertTrue(s.flush_cache(empty_cache=False))
        s.tree_cache.reset.assert_called_once_with()
        self.assertEqual(s.pin_table, {})
        # Nothing left to expire or unpin against the fresh tree.
        self.now = 1e9
        s.tree_cache.unlock = MagicMock()
        self.step(s)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertTrue(out.success)
        self.assertEqual(out.message, "request a was not pinned")
        s.tree_cache.unlock.assert_not_called()

    def test_role_switch_reset_clears_table(self):
        try:
            from sglang.srt.disaggregation.role_switch import (
                _release_prefix_cache_for_role_switch,
            )
        except ImportError as e:  # pragma: no cover
            self.skipTest(f"role_switch not importable: {e}")
        s = self.make_scheduler()
        self.add_req("a", 100)
        self.pin(s, "a", 5)
        s.disable_radix_cache = False
        s.tree_cache = MagicMock(wraps=self.cache)
        del s.tree_cache.clear_storage_backend
        s.req_to_token_pool = MagicMock()
        s.token_to_kv_pool_allocator = MagicMock()
        _release_prefix_cache_for_role_switch(s)
        s.tree_cache.reset.assert_called_once_with()
        self.assertEqual(s.pin_table, {})

    # ------------------------------------------------------------ flag off

    def test_flag_off(self):
        s = self.make_scheduler(enable=False)
        self.add_req("a", 100)
        out = self.pin(s, "a", 5)
        self.assertFalse(out.success)
        self.assertEqual(out.message, "--enable-joint-checkpoint is not set")
        self.assertIsNone(out.expires_in_s)
        out = s.unpin_request(UnpinReqInput(rid="a"))
        self.assertFalse(out.success)
        self.assertEqual(out.message, "--enable-joint-checkpoint is not set")
        self.assertEqual(s.pin_table, {})
        # flush_cache leaves the table alone without the flag (as before).
        s.pin_table = {"x": "sentinel"}
        self.flush_ready(s)
        self.assertTrue(s.flush_cache(empty_cache=False))
        self.assertEqual(s.pin_table, {"x": "sentinel"})


if __name__ == "__main__":
    unittest.main()
