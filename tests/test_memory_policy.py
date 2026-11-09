import ast
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest

from lyra.memory_policy import (
    DEFAULT_CAPACITY_SLOTS,
    DEFAULT_SLOT_BYTES,
    MemoryBudgetExceeded,
    MemoryLeasePool,
)
import lyra.memory_policy as memory_policy


class MemoryLeasePoolTests(unittest.TestCase):
    def test_default_policy_exposes_two_four_gib_slots(self):
        pool = MemoryLeasePool()

        self.assertEqual(DEFAULT_CAPACITY_SLOTS, 2)
        self.assertEqual(DEFAULT_SLOT_BYTES, 4 * 1024**3)
        self.assertEqual(pool.capacity_slots, 2)
        self.assertEqual(pool.slot_bytes, 4 * 1024**3)
        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, 2)
        with self.assertRaises(AttributeError):
            pool.capacity_slots = 3  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            pool.used_slots = 1  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            pool.available_slots = 1  # type: ignore[misc]

    def test_one_slot_and_two_slot_leases_account_for_ordinary_and_paired_jobs(self):
        ordinary_pool = MemoryLeasePool()
        ordinary = ordinary_pool.acquire("ordinary-job")

        self.assertEqual(ordinary.kind, "ordinary-job")
        self.assertEqual(ordinary.slots, 1)
        self.assertEqual(ordinary_pool.used_slots, 1)
        self.assertEqual(ordinary_pool.available_slots, 1)

        ordinary.close()
        self.assertEqual(ordinary_pool.used_slots, 0)

        paired_pool = MemoryLeasePool()
        paired = paired_pool.acquire("paired-band", slots=2)

        self.assertEqual(paired.kind, "paired-band")
        self.assertEqual(paired.slots, 2)
        self.assertEqual(paired_pool.used_slots, 2)
        self.assertEqual(paired_pool.available_slots, 0)

    def test_insufficient_capacity_raises_dedicated_error_without_mutation(self):
        pool = MemoryLeasePool()
        pool.acquire("ordinary-job")

        with self.assertRaises(MemoryBudgetExceeded):
            pool.acquire("paired-band", slots=2)

        self.assertEqual(pool.used_slots, 1)
        self.assertEqual(pool.available_slots, 1)

    def test_close_is_idempotent_and_context_management_releases_the_lease(self):
        pool = MemoryLeasePool()
        lease = pool.acquire("ordinary-job")

        lease.close()
        lease.close()
        self.assertEqual(pool.used_slots, 0)

        with pool.acquire("paired-band", slots=2) as paired:
            self.assertEqual(paired.kind, "paired-band")
            self.assertEqual(pool.used_slots, 2)

        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, 2)

    def test_context_manager_releases_and_propagates_body_exception(self):
        pool = MemoryLeasePool()

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with pool.acquire("ordinary-job"):
                raise RuntimeError("body failed")

        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, pool.capacity_slots)

    def test_invalid_arguments_reject_bool_slots_and_impossible_requests(self):
        pool = MemoryLeasePool()

        with self.assertRaises(TypeError):
            pool.acquire(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pool.acquire("")
        with self.assertRaises(TypeError):
            pool.acquire("ordinary-job", slots=True)
        with self.assertRaises(TypeError):
            pool.acquire("ordinary-job", slots=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            pool.acquire("ordinary-job", slots=0)
        with self.assertRaises(ValueError):
            pool.acquire("ordinary-job", slots=-1)
        with self.assertRaises(ValueError):
            pool.acquire("paired-band", slots=3)

        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, 2)

    def test_invalid_pool_configuration_is_rejected(self):
        with self.assertRaises(TypeError):
            MemoryLeasePool(capacity_slots=True)
        with self.assertRaises(TypeError):
            MemoryLeasePool(capacity_slots=2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MemoryLeasePool(capacity_slots=0)
        with self.assertRaises(TypeError):
            MemoryLeasePool(slot_bytes=True)
        with self.assertRaises(ValueError):
            MemoryLeasePool(slot_bytes=0)

    def test_concurrent_acquisitions_never_oversubscribe_capacity(self):
        pool = MemoryLeasePool()
        contender_count = 12
        with ThreadPoolExecutor(max_workers=12) as executor:
            for _ in range(25):
                start = threading.Barrier(contender_count + 1)

                def acquire_one() -> object:
                    start.wait()
                    try:
                        return pool.acquire("ordinary-job")
                    except MemoryBudgetExceeded as exc:
                        return exc

                pool._lock.acquire()
                try:
                    futures = [
                        executor.submit(acquire_one)
                        for _ in range(contender_count)
                    ]
                    start.wait()
                finally:
                    pool._lock.release()

                results = [future.result() for future in futures]
                leases = [
                    result
                    for result in results
                    if not isinstance(result, Exception)
                ]
                self.assertEqual(len(leases), pool.capacity_slots)
                self.assertEqual(pool.used_slots, pool.capacity_slots)
                self.assertEqual(pool.available_slots, 0)

                for lease in leases:
                    lease.close()
                self.assertEqual(pool.used_slots, 0)

    def test_concurrent_closes_release_a_lease_only_once(self):
        pool = MemoryLeasePool()
        lease = pool.acquire("paired-band", slots=2)
        start = threading.Barrier(16)

        def close_lease() -> None:
            start.wait()
            lease.close()

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(close_lease) for _ in range(16)]
            for future in futures:
                future.result()

        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, pool.capacity_slots)

    def test_closing_a_lease_changes_only_its_owning_pool(self):
        left_pool = MemoryLeasePool()
        right_pool = MemoryLeasePool()
        left_lease = left_pool.acquire("left", slots=2)
        right_pool.acquire("right")

        left_lease.close()

        self.assertEqual(left_pool.used_slots, 0)
        self.assertEqual(right_pool.used_slots, 1)

    def test_lease_ownership_and_metadata_are_immutable(self):
        owner = MemoryLeasePool()
        other = MemoryLeasePool()
        lease = owner.acquire("ordinary-job")

        with self.assertRaises(AttributeError):
            lease.kind = "paired-band"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            lease.slots = 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            lease._pool = other  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            lease._token = object()  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            lease._slots = 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            del lease._pool  # type: ignore[attr-defined]
        with self.assertRaises(AttributeError):
            del lease._slots  # type: ignore[attr-defined]

        lease.close()
        self.assertEqual(lease.kind, "ordinary-job")
        self.assertEqual(lease.slots, 1)
        self.assertEqual(owner.used_slots, 0)
        self.assertEqual(other.used_slots, 0)

    def test_release_derives_slots_from_pool_registry_not_lease_fields(self):
        pool = MemoryLeasePool()
        lease = pool.acquire("ordinary-job")
        object.__setattr__(lease, "_slots", 2)

        lease.close()

        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, pool.capacity_slots)

    def test_unknown_foreign_and_duplicate_tokens_cannot_debit_pools(self):
        owner = MemoryLeasePool()
        foreign = MemoryLeasePool()
        lease = owner.acquire("paired-band", slots=2)
        (token,) = owner._active_leases

        self.assertIs(type(token), object)
        self.assertFalse(foreign._release(token))
        self.assertFalse(owner._release(object()))
        self.assertEqual(owner.used_slots, 2)
        self.assertEqual(foreign.used_slots, 0)

        self.assertTrue(owner._release(token))
        self.assertFalse(owner._release(token))
        lease.close()

        for pool in (owner, foreign):
            self.assertGreaterEqual(pool.used_slots, 0)
            self.assertLessEqual(pool.used_slots, pool.capacity_slots)
            self.assertEqual(
                pool.available_slots,
                pool.capacity_slots - pool.used_slots,
            )

    def test_concurrent_duplicate_token_release_retires_exactly_once(self):
        pool = MemoryLeasePool()
        pool.acquire("paired-band", slots=2)
        (token,) = pool._active_leases
        start = threading.Barrier(16)

        def release_token() -> bool:
            start.wait()
            return pool._release(token)

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(lambda _index: release_token(), range(16)))

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 15)
        self.assertEqual(pool.used_slots, 0)
        self.assertEqual(pool.available_slots, pool.capacity_slots)

    def test_excessive_registry_release_is_rejected_without_mutation(self):
        pool = MemoryLeasePool()
        pool.acquire("ordinary-job")
        (token,) = pool._active_leases
        pool._active_leases[token] = pool.capacity_slots + 1

        with self.assertRaisesRegex(RuntimeError, "invariant"):
            pool._release(token)

        self.assertEqual(pool.used_slots, 1)
        self.assertEqual(pool.available_slots, 1)
        self.assertIn(token, pool._active_leases)

    def test_module_has_no_cache_or_filesystem_dependencies(self):
        tree = ast.parse(inspect.getsource(memory_policy))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        forbidden_call_names = {"__import__", "open"}
        filesystem_method_names = {
            "glob",
            "iterdir",
            "mkdir",
            "open",
            "read_bytes",
            "read_text",
            "rename",
            "replace",
            "rmdir",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        called_names = {call.id for call in calls if isinstance(call, ast.Name)}
        called_methods = {
            call.attr for call in calls if isinstance(call, ast.Attribute)
        }

        self.assertEqual(imported_modules, {"threading"})
        self.assertTrue(called_names.isdisjoint(forbidden_call_names))
        self.assertTrue(called_methods.isdisjoint(filesystem_method_names))


if __name__ == "__main__":
    unittest.main()
