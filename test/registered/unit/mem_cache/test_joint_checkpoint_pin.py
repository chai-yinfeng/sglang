"""Joint checkpoint primitives on UnifiedRadixCache, CPU only.

pin   = checkpoint(req, up_to) + an extra TreeLock held outside the request
        (scheduler.pin_request); the prefix must survive eviction after the
        request's own lock is gone, and become evictable again after unlock.
discard = release_kv_cache(checkpoint=True) with req.skip_radix_cache_insert
        set (scheduler.abort_request with AbortReq.discard); nothing of the
        request may land in the tree.
"""

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

POOL_SIZE = 64


def make_cache(page_size=1) -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=page_size,
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
        size=8,
        max_context_len=128,
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
        )
    )


def insert(cache, token_ids):
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node


def match_len(cache, token_ids) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).device_prefix_len


class MockReq:
    """The Req fields checkpoint / release_kv_cache read, for a request that
    matched nothing and has `prompt_len` prompt tokens plus generated tokens."""

    rid = "mock-req"
    session = None
    session_id = None
    extra_key = None
    cache_salt = None
    priority = 0
    kv_rotation_base = None
    finished_reason = None
    skip_radix_cache_insert = False
    finished_len = None

    def __init__(self, cache, fill_ids, prompt_len):
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


class TestJointCheckpointPin(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.cache = make_cache()

    def test_extra_lock_survives_eviction_until_unlocked(self):
        a = list(range(1, 9))
        b = list(range(21, 29))
        node_a = insert(self.cache, a)
        insert(self.cache, b)
        self.assertEqual(self.cache.evictable_size(), 16)

        pin = self.cache.lock(node_a)
        self.assertEqual(self.cache.evictable_size(), 8)
        self.cache.evict(EvictParams(num_tokens=16))
        self.assertEqual(match_len(self.cache, a), 8)
        self.assertEqual(match_len(self.cache, b), 0)

        self.cache.unlock(pin)
        self.assertEqual(self.cache.evictable_size(), 8)
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(match_len(self.cache, a), 0)
        self.assertEqual(self.cache.evictable_size(), 0)

    def test_pin_running_req_then_finish_keeps_prefix_drops_tail(self):
        """scheduler.pin_request flow: checkpoint(up_to) on a running request,
        extra lock on its last_node, then the request finishes (abort writes
        the rest back) and its own lock goes away."""
        prompt = list(range(100, 110))  # 10 prompt tokens
        fill = prompt + list(range(200, 206))  # + 6 generated
        req = MockReq(self.cache, fill, prompt_len=len(prompt))
        pin_up_to = 12  # prompt + 2 generated

        self.cache.checkpoint(req, up_to=pin_up_to)
        self.assertEqual(req.kv.cache_protected_len, pin_up_to)
        pin = self.cache.lock(req.last_node)

        req.finished_reason = "abort"
        release_kv_cache(req, self.cache, checkpoint=True)
        self.assertEqual(match_len(self.cache, fill), len(fill))
        # Only the tail past the pin is evictable.
        self.assertEqual(self.cache.evictable_size(), len(fill) - pin_up_to)

        self.cache.evict(EvictParams(num_tokens=len(fill)))
        self.assertEqual(match_len(self.cache, fill), pin_up_to)
        self.assertEqual(self.cache.evictable_size(), 0)

        self.cache.unlock(pin)
        self.cache.evict(EvictParams(num_tokens=len(fill)))
        self.assertEqual(match_len(self.cache, fill), 0)
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.available_size(), POOL_SIZE
        )

    def test_pin_shorter_than_protected_prefix_holds_last_node(self):
        """up_to below cache_protected_len: no checkpoint, the matched prefix
        node is held (pins a superset of what was asked)."""
        shared = list(range(1, 9))
        insert(self.cache, shared)
        fill = shared + [50, 51]
        req = MockReq(self.cache, fill, prompt_len=len(shared))
        self.assertEqual(req.kv.cache_protected_len, 0)
        req.kv.cache_protected_len = 8  # as match_prefix would have set it
        pin = self.cache.lock(req.last_node)
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(match_len(self.cache, shared), 8)
        self.cache.unlock(pin)
        self.cache.unlock(req.lock)
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(match_len(self.cache, shared), 0)

    def test_discard_releases_kv_without_tree_insert(self):
        prompt = list(range(300, 310))
        fill = prompt + [400, 401, 402]
        req = MockReq(self.cache, fill, prompt_len=len(prompt))
        req.skip_radix_cache_insert = True
        req.finished_reason = "abort"
        release_kv_cache(req, self.cache, checkpoint=True)

        self.assertEqual(match_len(self.cache, fill), 0)
        self.assertEqual(self.cache.evictable_size(), 0)
        self.assertEqual(
            self.cache.token_to_kv_pool_allocator.available_size(), POOL_SIZE
        )
        self.assertFalse(req.kv.holds_kv)

    def test_plain_abort_still_inserts(self):
        prompt = list(range(300, 310))
        fill = prompt + [400, 401, 402]
        req = MockReq(self.cache, fill, prompt_len=len(prompt))
        req.finished_reason = "abort"
        release_kv_cache(req, self.cache, checkpoint=True)
        self.assertEqual(match_len(self.cache, fill), len(fill))
        self.assertEqual(self.cache.evictable_size(), len(fill))


if __name__ == "__main__":
    unittest.main()
