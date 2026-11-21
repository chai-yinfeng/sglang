"""Joint checkpoint on a hybrid (mamba) model, CPU only: a Scheduler built
with Scheduler.__new__ over a real UnifiedRadixCache with FULL + MAMBA
components, a real HybridReqToTokenPool (extra buffer, two track entries) and
real Req objects.

A recurrent state only returns to positions where it was saved, so every
hold ends on a node whose mamba value is the state exactly at its end, and
pinned_len / jc_state_at / jc_truncated_at report that position:

- pin dispatch: a position decode has not reached is registered and pinned
  once decode passes it; the head without overlap is copied from the working
  slot; an earlier position lands on the nearest saved state (tree, history
  ring or track buffer), recomputing the rest, never more than it holds;
- the snapshot history keeps the states the track buffer overwrites, at most
  --jc-mamba-history-slots of them, degrades to nothing when the mamba pool
  is full, gives its slots back on release and to eviction under pressure;
- truncate cuts at the nearest saved state at or before pos, also below the
  request's protected prefix, and refuses when there is none;
- the pin cap weighs each held state as jc_mamba_snapshot_tokens tokens.

Slot contents are tagged with the position they stand for (the temporal
state of every layer is filled with it), so a test can tell which state a
tree node ended up with.
"""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PinReqInput, TruncateParams, UnpinReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.jc_mamba_history import (
    JcMambaHistory,
    JcMambaState,
    classify_pin_target,
    decode_track_masks,
    hybrid_config_error,
    select_mamba_states,
    snapshot_token_equivalent,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache import fixtures as urc_fixtures
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KV_SIZE = 256
MAMBA_SLOTS = 24
PROMPT = list(range(100, 110))  # 10 prompt tokens
OUTPUT = list(range(200, 220))  # 20 generated tokens
FILL = PROMPT + OUTPUT


class TestJcMambaSelection(CustomTestCase):
    """The pure choices: where a pin's state comes from, which saved state a
    position lands on, which decode steps track, what a state weighs."""

    def test_pin_dispatch_against_the_head(self):
        def kind(target, overlap=True, head_tracked=False):
            return classify_pin_target(
                target=target, head=30, overlap=overlap, head_tracked=head_tracked
            )

        # Decode has not produced it: always a capture, with or without overlap.
        self.assertEqual(kind(31), "capture")
        self.assertEqual(kind(64, overlap=False), "capture")
        # The head: in the working slot without overlap; under overlap only
        # the in-flight step's own track can give it.
        self.assertEqual(kind(30, overlap=False), "copy")
        self.assertEqual(kind(30, head_tracked=True), "capture")
        self.assertEqual(kind(30), "past")
        # Before the head: a saved state, even when the head step tracks.
        self.assertEqual(kind(29, overlap=False), "past")
        self.assertEqual(kind(29, head_tracked=True), "past")

    def test_nearest_state_at_or_before_the_target(self):
        states = [
            JcMambaState(seqlen=24, source="track", track_idx=0),
            JcMambaState(seqlen=12, source="ring"),
            JcMambaState(seqlen=18, source="track", track_idx=1),
            JcMambaState(seqlen=12, source="tree", node=7),
        ]
        self.assertEqual(
            [s.seqlen for s in select_mamba_states(states, 23)], [18, 12, 12]
        )
        self.assertEqual(select_mamba_states(states, 24)[0].seqlen, 24)
        # A tie goes to the tree: nothing to publish.
        self.assertEqual(select_mamba_states(states, 17)[0].source, "tree")
        self.assertEqual(select_mamba_states(states, 11), [])

    def test_capture_points_join_the_track_masks(self):
        mask, mask_next = decode_track_masks(
            seq_lens=[256, 37, 37, 255, 40],
            capture_at=[None, 37, 38, None, 37],
            track_interval=256,
        )
        self.assertEqual(mask, [True, True, False, False, False])
        # The step after: an overlap scheduler must know it tracks.
        self.assertEqual(mask_next, [False, False, True, True, False])

    def test_snapshot_token_equivalent_rounds_up(self):
        # Qwen3.8-27B (docs/design-hybrid-truncate.md section 2): a 147 MiB
        # fp32 snapshot over 64 KiB of full-attention KV per token.
        self.assertEqual(
            snapshot_token_equivalent(
                snapshot_bytes=147 << 20, kv_bytes_per_token=64 << 10
            ),
            2352,
        )
        self.assertEqual(
            snapshot_token_equivalent(snapshot_bytes=65, kv_bytes_per_token=64), 2
        )

    def test_only_the_eager_extra_buffer_is_supported(self):
        ok = dict(extra_buffer=True, lazy=False, int8_checkpoint=False, replayssm=False)
        self.assertIsNone(hybrid_config_error(**ok))
        for field, needle in (
            ("extra_buffer", "extra_buffer"),
            ("lazy", "extra_buffer_lazy"),
            ("int8_checkpoint", "int8"),
            ("replayssm", "replayssm"),
        ):
            bad = dict(ok, **{field: not ok[field]})
            self.assertIn(needle, hybrid_config_error(**bad))


class _HybridFixture(CustomTestCase):
    history_slots = 4

    def setUp(self):
        super().setUp()
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            mamba_cache_size=MAMBA_SLOTS,
            kv_size=KV_SIZE,
            max_num_reqs=6,
        )
        with patch.object(urc_fixtures, "get_device", return_value="cpu"):
            self.cache, self.allocator, self.pool = urc_fixtures.build_fixture(
                cfg, tree_core_backend="python"
            )
        self.history = JcMambaHistory(
            capacity=self.history_slots, allocator=self.pool.mamba_allocator
        )
        self.pool.jc_mamba_history = self.history
        self.mamba_free0 = self.pool.mamba_allocator.available_size()
        self.now = 1000.0
        clock = SimpleNamespace(monotonic=lambda: self.now)
        patcher = patch.object(scheduler_module, "time", clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ------------------------------------------------------------ fixtures

    def make_scheduler(
        self, running=(), overlap=True, max_ratio=0.5, overflow="refuse", tokens=0
    ):
        s = Scheduler.__new__(Scheduler)
        s.enable_joint_checkpoint = True
        s.jc_hybrid = True
        s.jc_mamba_snapshot_tokens = tokens
        s.jc_mamba_captures = {}
        s.enable_overlap = overlap
        s.disaggregation_mode = DisaggregationMode.NULL
        s.dllm_config = None
        s.page_size = 1
        s.pin_table = {}
        s.jc_pin_ttl_s = 600.0
        s.jc_pin_max_ttl_s = 3600.0
        s.jc_pin_max_ratio = max_ratio
        s.jc_pin_overflow = overflow
        s.jc_degraded_pins = {}
        s.jc_pins_degraded_total = 0
        s.jc_pins_repinned_total = 0
        s.max_total_num_tokens = KV_SIZE
        s.paused_reqs = {}
        s.chunked_req = None
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.tree_cache = self.cache
        s.req_to_token_pool = self.pool
        s.token_to_kv_pool_allocator = self.allocator
        s.running_batch = ScheduleBatch(reqs=list(running), batch_is_full=False)
        s.last_batch = None
        s.result_queue = deque()
        s.enable_continuous_input_polling = False
        s.waiting_queue = []
        s.draft_worker = None
        s.output_streamer = MagicMock()
        return s

    def start(self, rid="old", outputs=OUTPUT, overlap=True) -> Req:
        """A request prefilled from an empty tree and decoded ``outputs``;
        under overlap the step for its last output token is in flight
        (kv_committed_len == len(fill)), without it that token has no KV."""
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", PROMPT),
            sampling_params=SamplingParams(max_new_tokens=100),
        )
        req.init_next_round_input(self.cache)
        req.lock = self.cache.lock(req.last_node)
        self.pool.alloc([req])
        committed = len(PROMPT) + len(outputs) - (0 if overlap else 1)
        indices = self.allocator.alloc(committed)
        self.pool.write((req.kv.req_pool_idx, slice(0, committed)), indices)
        req.kv.kv_committed_len = committed
        req.kv.kv_allocated_len = committed
        req.output_ids = array("q", outputs)
        req.refresh_fill_ids()
        self.tag(req.kv.mamba_pool_idx, committed)
        return req

    def tag(self, slot, seqlen):
        self.pool.mamba_pool.mamba_cache.temporal[:, int(slot)] = seqlen

    def tag_of(self, slot) -> float:
        return float(
            self.pool.mamba_pool.mamba_cache.temporal[0, int(slot)].flatten()[0]
        )

    def track(self, req, seqlen):
        """A decode step reaching ``seqlen`` with its track mask set: the
        history retires the entry it writes before the forward, the forward
        writes the state there, and result processing names it."""
        self.history.retire_track_slots(reqs=[req], track_mask=[True], pool=self.pool)
        kv = req.kv
        idx = kv.mamba_next_track_idx
        self.tag(kv.mamba_ping_pong_track_buffer[idx], seqlen)
        kv.mamba_last_track_idx = idx
        kv.mamba_prev_track_seqlen = kv.mamba_last_track_seqlen
        kv.mamba_last_track_seqlen = seqlen
        kv.mamba_next_track_idx = 1 - idx

    def ring(self, req):
        return [snap.seqlen for snap in self.history.snapshots(req.kv.req_pool_idx)]

    def match(self, n):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", FILL[:n])))
        )

    def node_state_tag(self, node):
        value = self.cache.tree_core.get_component_device_value(
            node, ComponentType.MAMBA
        )
        return None if value is None else self.tag_of(value.flatten()[0])

    def pin(self, s, up_to, rid="old"):
        return s.pin_request(PinReqInput(rid=rid, up_to=up_to))

    def truncate(self, s, pos, rid="succ", old_rid="old"):
        succ = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", FILL[:pos] + [900, 901]),
            sampling_params=SamplingParams(max_new_tokens=100),
        )
        recv = SimpleNamespace(
            rid=rid, truncate=TruncateParams(rid=old_rid, pos=pos), session_params=None
        )
        return succ, s._jc_truncate(succ, recv)

    def assert_no_mamba_leak(self, s):
        """With every hold dropped, every request released and the tree
        evicted, the mamba pool is whole again."""
        for rid in list(s.pin_table):
            s.unpin_request(UnpinReqInput(rid=rid))
        self.cache.evict(EvictParams(num_tokens=KV_SIZE, mamba_num=MAMBA_SLOTS))
        self.assertEqual(self.history.num_slots, 0)
        self.assertEqual(self.pool.mamba_allocator.available_size(), self.mamba_free0)

    def finish(self, req):
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        req.finished_len = len(req.output_ids)
        release_kv_cache(req, self.cache, checkpoint=True)


class TestJcMambaHistory(_HybridFixture):
    history_slots = 2

    def test_ring_keeps_the_last_overwritten_states(self):
        req = self.start()
        for seqlen in (12, 16, 20, 24, 28):
            self.track(req, seqlen)
        # 28 and 24 are named by the track buffer; the ring holds what it
        # overwrote, the newest two of them.
        self.assertEqual(self.ring(req), [16, 20])
        self.assertEqual(
            (req.kv.mamba_last_track_seqlen, req.kv.mamba_prev_track_seqlen), (28, 24)
        )
        for snap in self.history.snapshots(req.kv.req_pool_idx):
            self.assertEqual(self.tag_of(snap.slot), snap.seqlen)
        # Working slot + 2 track entries + 2 ring snapshots.
        self.assertEqual(
            self.pool.mamba_allocator.available_size(), self.mamba_free0 - 5
        )

    def test_full_pool_drops_the_snapshot_and_its_name(self):
        req = self.start()
        for seqlen in (12, 16):
            self.track(req, seqlen)
        held = self.pool.mamba_allocator.alloc(
            self.pool.mamba_allocator.available_size()
        )
        self.track(req, 20)
        # 12 was overwritten in place: nothing names it any more.
        self.assertEqual(self.ring(req), [])
        self.assertEqual(self.history.alloc_failures, 1)
        self.assertEqual(
            (req.kv.mamba_last_track_seqlen, req.kv.mamba_prev_track_seqlen), (20, 16)
        )
        self.pool.mamba_allocator.free(held)

    def test_release_frees_the_ring(self):
        req = self.start()
        for seqlen in (12, 16, 20, 24):
            self.track(req, seqlen)
        self.assertEqual(self.history.num_slots, 2)
        self.finish(req)
        s = self.make_scheduler()
        self.assert_no_mamba_leak(s)

    def test_decode_step_tracks_a_capture_point_off_the_grid(self):
        """ScheduleBatch.prepare_for_decode's joint-checkpoint hook: a row
        whose capture point is this step tracks, and the entry it writes is
        retired into the ring first; other rows are left alone."""
        a, b = self.start("a"), self.start("b")
        for req in (a, b):
            self.track(req, 12)
            self.track(req, 18)
        a.jc_mamba_capture_at = 31
        batch = ScheduleBatch(reqs=[a, b], batch_is_full=False)
        batch.req_to_token_pool = self.pool
        batch.seq_lens_cpu = torch.tensor([31, 31])
        mask, mask_next = batch._jc_prepare_decode_track(256)
        self.assertEqual((mask, mask_next), ([True, False], [False, False]))
        self.assertEqual((self.ring(a), self.ring(b)), ([12], []))
        self.assertEqual(b.kv.mamba_prev_track_seqlen, 12)

    def test_eviction_for_an_allocation_reclaims_the_oldest_snapshots(self):
        a, b = self.start("a"), self.start("b")
        for seqlen in (12, 16, 20):
            self.track(a, seqlen)
            self.track(b, seqlen)
        self.track(a, 24)
        self.assertEqual((self.ring(a), self.ring(b)), ([12, 16], [12]))
        free = self.pool.mamba_allocator.available_size()
        # Two slots short, with nothing in the tree to evict.
        self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=2))
        # a's 12 and b's 12 went first; a's 16 stays.
        self.assertEqual((self.ring(a), self.ring(b)), ([16], []))
        self.assertEqual(self.pool.mamba_allocator.available_size(), free + 2)


class TestJcHybridPin(_HybridFixture):
    def setUp(self):
        super().setUp()
        self.req = self.start()
        for seqlen in (12, 18, 24):
            self.track(self.req, seqlen)
        # ring [12], track buffer 24 (latest) and 18; head 30.
        self.s = self.make_scheduler([self.req])

    def assert_holds_state(self, out, at):
        """The hold ends on a node whose mamba value is the state at ``at``."""
        self.assertTrue(out.success, out.message)
        self.assertEqual((out.pinned_len, out.jc_state_at), (at, at))
        pin = self.s.pin_table["old"]
        self.assertEqual(self.node_state_tag(pin.lock.node), at)
        self.assertEqual(self.match(at).device_prefix_len, at)

    def test_earlier_position_lands_on_the_nearest_saved_state(self):
        out = self.pin(self.s, 22)
        self.assert_holds_state(out, 18)
        self.assertIn("recomputes [18, 22)", out.message)
        # The track entry was handed to the tree and stops naming 18.
        self.assertIsNone(self.req.kv.mamba_prev_track_seqlen)
        self.assertEqual(self.req.kv.mamba_last_track_seqlen, 24)

    def test_ring_state_moves_into_the_tree(self):
        out = self.pin(self.s, 13)
        self.assert_holds_state(out, 12)
        self.assertEqual(self.ring(self.req), [])
        self.assertEqual(out.mamba_history_slots, 0)

    def test_state_below_the_protected_prefix_is_attached(self):
        self.assert_holds_state(self.pin(self.s, 24), 24)
        self.assertEqual(self.req.kv.cache_protected_len, 24)
        # Re-pin earlier: the ring's 12 joins the published path at 12.
        self.assert_holds_state(self.pin(self.s, 15), 12)
        self.assertEqual(self.req.kv.cache_protected_len, 24)
        self.assertEqual(self.node_state_tag(self.match(24).last_device_node), 24)

    def test_head_without_overlap_copies_the_working_slot(self):
        req = self.start("b", overlap=False)
        s = self.make_scheduler([req], overlap=False)
        head = req.kv.kv_committed_len
        out = self.pin(s, head, rid="b")
        self.assertTrue(out.success, out.message)
        self.assertEqual((out.pinned_len, out.jc_state_at), (head, head))
        self.assertEqual(self.node_state_tag(s.pin_table["b"].lock.node), head)
        # A copy: the working slot is still the request's own.
        self.assertEqual(self.tag_of(req.kv.mamba_pool_idx), head)

    def test_head_under_overlap_is_not_claimed_exact(self):
        out = self.pin(self.s, 30)
        self.assert_holds_state(out, 24)

    def test_future_position_is_registered_then_pinned(self):
        out = self.pin(self.s, 33)
        self.assertTrue(out.success)
        self.assertEqual(
            (out.jc_capture_at, out.pinned_len, out.jc_state_at), (33, 0, None)
        )
        self.assertIn("takes effect then", out.message)
        self.assertNotIn("old", self.s.pin_table)
        self.assertEqual(self.req.jc_mamba_capture_at, 33)
        # Decode has launched the step reaching 33 but not processed it.
        self.track(self.req, 33)
        self.s._jc_resolve_mamba_captures()
        self.assertNotIn("old", self.s.pin_table)
        # Its result is in: the token after 33 was sampled.
        self.req.output_ids = array(
            "q", list(self.req.output_ids) + [300, 301, 302, 303]
        )
        self.s._jc_resolve_mamba_captures()
        self.assertEqual(self.s.jc_mamba_captures, {})
        self.assertIsNone(self.req.jc_mamba_capture_at)
        pin = self.s.pin_table["old"]
        self.assertEqual(pin.pinned_len, 33)
        self.assertEqual(self.node_state_tag(pin.lock.node), 33)

    def test_lost_capture_pins_the_state_before_it(self):
        self.pin(self.s, 33)
        self.req.output_ids = array(
            "q", list(self.req.output_ids) + [300, 301, 302, 303]
        )
        self.s._jc_resolve_mamba_captures()
        self.assertEqual(self.s.pin_table["old"].pinned_len, 24)

    def test_capture_of_a_request_that_finishes_first_is_dropped(self):
        self.pin(self.s, 40)
        self.s.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.s._jc_resolve_mamba_captures()
        self.assertEqual(self.s.jc_mamba_captures, {})
        self.assertEqual(self.s.pin_table, {})

    def test_no_saved_state_refuses(self):
        out = self.pin(self.s, 11)
        self.assertFalse(out.success)
        self.assertIn("no saved recurrent", out.message)
        self.assertEqual(self.s.pin_table, {})
        self.assertEqual(self.match(30).device_prefix_len, 0)

    def test_snapshot_weighs_its_token_equivalent_in_the_cap(self):
        # The hold's 18 KV tokens fit under the cap of 32; 18 + 20 do not.
        s = self.make_scheduler([self.req], max_ratio=32 / KV_SIZE, tokens=20)
        out = self.pin(s, 18)
        self.assertFalse(out.success)
        self.assertIn("pin cap exceeded", out.message)
        s.jc_mamba_snapshot_tokens = 14
        out = self.pin(s, 18)
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.pinned_tokens_total, 18 + 14)

    def test_finish_after_a_pin_past_the_latest_track(self):
        """The head copy publishes a state past the latest decode track (24);
        finishing then has nothing longer to insert and keeps the published
        prefix instead of asserting on a shorter one."""
        s = self.make_scheduler([self.req], overlap=False)
        head = self.req.kv.kv_committed_len
        out = self.pin(s, head)
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.jc_state_at, head)
        self.assertEqual(self.req.kv.cache_protected_len, head)
        self.finish(self.req)
        self.assertEqual(self.match(head).device_prefix_len, head)
        self.assertEqual(self.node_state_tag(s.pin_table["old"].lock.node), head)
        self.assert_no_mamba_leak(s)

    def test_no_mamba_slot_leaks(self):
        self.assertTrue(self.pin(self.s, 13).success)
        self.assertTrue(self.pin(self.s, 22).success)
        self.finish(self.req)
        self.assert_no_mamba_leak(self.s)


class TestJcHybridTruncate(_HybridFixture):
    def setUp(self):
        super().setUp()
        self.req = self.start()
        for seqlen in (12, 18, 24):
            self.track(self.req, seqlen)
        self.s = self.make_scheduler([self.req])

    def test_inflight_cut_lands_on_the_nearest_saved_state(self):
        succ, err = self.truncate(self.s, 22)
        self.assertIsNone(err)
        self.assertEqual((succ.jc_truncated_at, succ.jc_state_at), (18, 18))
        self.assertTrue(self.req.finished())
        hold = self.s.pin_table["succ"]
        self.assertEqual(hold.pinned_len, 18)
        self.assertEqual(self.node_state_tag(hold.lock.node), 18)
        # The successor's match stops there and recomputes [18, 22).
        succ.init_next_round_input(self.cache)
        self.assertEqual(succ.prefix_len, 18)

    def test_cut_below_a_pin_attaches_the_ring_state(self):
        self.assertTrue(self.pin(self.s, 24).success)
        succ, err = self.truncate(self.s, 15)
        self.assertIsNone(err)
        self.assertEqual(succ.jc_state_at, 12)
        self.assertEqual(self.node_state_tag(self.s.pin_table["succ"].lock.node), 12)

    def test_no_saved_state_refuses_and_leaves_the_request(self):
        succ, err = self.truncate(self.s, 11)
        self.assertIn("no saved recurrent", err)
        self.assertFalse(self.req.finished())
        self.assertEqual(self.s.pin_table, {})
        self.assertEqual(self.ring(self.req), [12])

    def test_finished_request_cuts_at_a_state_inside_its_pin(self):
        self.assertTrue(self.pin(self.s, 22).success)  # state at 18, ring keeps 12
        self.assertTrue(self.pin(self.s, 13, rid="old").success)  # 12 on the path
        self.assertTrue(self.pin(self.s, 24).success)  # hold moves to 24
        self.finish(self.req)
        self.s.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        succ, err = self.truncate(self.s, 21)
        self.assertIsNone(err)
        self.assertEqual(succ.jc_state_at, 18)
        self.assertEqual(self.node_state_tag(self.s.pin_table["succ"].lock.node), 18)
        succ, err = self.truncate(self.s, 11, rid="succ2")
        self.assertIn("no saved recurrent", err)

    def test_no_mamba_slot_leaks(self):
        succ, err = self.truncate(self.s, 22)
        self.assertIsNone(err)
        self.s._jc_release_truncate_hold(succ, "aborted")
        self.assert_no_mamba_leak(self.s)


if __name__ == "__main__":
    unittest.main()
