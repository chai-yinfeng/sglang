"""Joint checkpoint P4: UnifiedRadixCache.raise_path_priority on a CPU tree.

scheduler.set_priority calls it when a request's priority goes up: every
node on the root-to-last_node path takes max(node.priority, new), as insert
does, so the `priority` eviction policy keeps the path longer. A lower value
is never written (the scheduler does not call it), and a smaller argument
leaves the nodes alone.
"""

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

POOL_SIZE = 64


def make_cache(tree_core_backend, eviction_policy="priority") -> UnifiedRadixCache:
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
            page_size=1,
            eviction_policy=eviction_policy,
            tree_components=(ComponentType.FULL,),
            tree_core_backend=tree_core_backend,
        )
    )


def insert(cache, token_ids, priority=0):
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
            priority=priority,
        )
    ).last_device_node


def match_len(cache, token_ids) -> int:
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", token_ids)))
    ).device_prefix_len


def path_priorities(cache, node_id):
    """Priorities on the path from node_id up to (excluding) the root."""
    core = cache.tree_core
    out = []
    node = core.node_by_id(node_id)
    while node is not None and node is not core.root_node:
        out.append(node.priority)
        node = node.parent
    return out


class TestJointCheckpointPriorityTree(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def test_raise_updates_path_and_lower_does_not(self):
        cache = make_cache("python")
        self.assertIsInstance(cache.tree_core, UnifiedTreeCore)
        shared = list(range(1, 5))
        a = shared + list(range(11, 15))
        b = shared + list(range(21, 25))
        node_a = insert(cache, a)
        node_b = insert(cache, b)
        # Two nodes on each path (shared prefix + own tail), all at 0.
        self.assertEqual(path_priorities(cache, node_a), [0, 0])
        self.assertEqual(path_priorities(cache, node_b), [0, 0])

        self.assertTrue(cache.raise_path_priority(node_a, 5))
        self.assertEqual(path_priorities(cache, node_a), [5, 5])
        # b shares the prefix node, so it sees the raise there and only there.
        self.assertEqual(path_priorities(cache, node_b), [0, 5])

        # A smaller value is a no-op on existing nodes.
        self.assertTrue(cache.raise_path_priority(node_a, 1))
        self.assertEqual(path_priorities(cache, node_a), [5, 5])

    def test_raised_path_survives_priority_eviction(self):
        cache = make_cache("python")
        a = list(range(1, 9))
        b = list(range(21, 29))
        node_a = insert(cache, a)
        insert(cache, b)
        # Same priority and a is older: LRU would take a first.
        cache.raise_path_priority(node_a, 5)
        cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(match_len(cache, a), 8)
        self.assertEqual(match_len(cache, b), 0)

    def test_rust_core_reports_not_applied(self):
        cache = make_cache("rust")
        if not isinstance(cache.tree_core, RustUnifiedTreeCore):
            self.skipTest("Rust TreeCore not available; fell back to Python")
        node_a = insert(cache, list(range(1, 9)))
        self.assertFalse(cache.raise_path_priority(node_a, 5))


if __name__ == "__main__":
    unittest.main()
