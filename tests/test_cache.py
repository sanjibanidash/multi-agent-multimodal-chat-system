"""Tests for the in-memory cache manager."""
import time
import pytest
from backend.cache.cache_manager import CacheManager


@pytest.fixture
def cache():
    """Fresh cache with a short TTL for testing."""
    return CacheManager(max_size=5, ttl=2)  # 2-second TTL


def test_cache_miss_on_empty(cache):
    hit, val = cache.get("nonexistent-key")
    assert hit is False
    assert val is None


def test_cache_hit_after_set(cache):
    cache.set("key1", {"answer": "Paris"})
    hit, val = cache.get("key1")
    assert hit is True
    assert val == {"answer": "Paris"}


def test_cache_miss_after_ttl_expires(cache):
    cache.set("key2", "some value")
    time.sleep(2.1)  # Wait for TTL to expire
    hit, val = cache.get("key2")
    assert hit is False


def test_cache_evicts_oldest_when_full(cache):
    for i in range(5):
        cache.set(f"key{i}", f"value{i}")
    # Adding a 6th entry should evict 'key0'
    cache.set("key5", "value5")
    hit, _ = cache.get("key0")
    assert hit is False


def test_make_key_is_deterministic():
    k1 = CacheManager.make_key("sql", "What is total sales?")
    k2 = CacheManager.make_key("sql", "What is total sales?")
    assert k1 == k2


def test_make_key_normalizes_whitespace():
    k1 = CacheManager.make_key("sql", "total  sales")
    k2 = CacheManager.make_key("sql", "total sales")
    assert k1 == k2


def test_make_key_differs_by_agent():
    k1 = CacheManager.make_key("sql", "hello")
    k2 = CacheManager.make_key("rag", "hello")
    assert k1 != k2
