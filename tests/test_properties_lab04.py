"""Property based invariants for postings and merging."""
from hypothesis import given, strategies as st
from findex.search import merge_intersection, merge_union, merge_difference

lists = st.lists(st.integers(min_value=0, max_value=100), max_size=40).map(
    lambda items: sorted(set(items))
)

@given(lists, lists)
def test_merge_matches_sets(a, b):
    assert merge_intersection(a, b) == sorted(set(a) & set(b))
    assert merge_union(a, b) == sorted(set(a) | set(b))
    assert merge_difference(a, b) == sorted(set(a) - set(b))

@given(lists)
def test_sorted_unique(items):
    assert items == sorted(set(items))
