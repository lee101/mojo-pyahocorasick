"""Behavioral parity with the pyahocorasick 2.x extension."""

from __future__ import annotations

import pickle
import random

import ahocorasick as upstream
import pytest

import mojo_pyahocorasick as mojo


CONSTANTS = [
    "EMPTY",
    "TRIE",
    "AHOCORASICK",
    "STORE_ANY",
    "STORE_INTS",
    "STORE_LENGTH",
    "KEY_STRING",
    "KEY_SEQUENCE",
    "MATCH_EXACT_LENGTH",
    "MATCH_AT_MOST_PREFIX",
    "MATCH_AT_LEAST_PREFIX",
]


def paired(words, store=mojo.STORE_ANY, key_type=mojo.KEY_STRING):
    ours = mojo.Automaton(store, key_type)
    theirs = upstream.Automaton(store, key_type)
    for key, value in words:
        if store == mojo.STORE_LENGTH:
            assert ours.add_word(key) == theirs.add_word(key)
        else:
            assert ours.add_word(key, value) == theirs.add_word(key, value)
    return ours, theirs


def compile_pair(words, store=mojo.STORE_ANY, key_type=mojo.KEY_STRING):
    ours, theirs = paired(words, store, key_type)
    assert ours.make_automaton() == theirs.make_automaton()
    return ours, theirs


def test_constants_match_upstream():
    assert {name: getattr(mojo, name) for name in CONSTANTS} == {
        name: getattr(upstream, name) for name in CONSTANTS
    }


def test_canonical_aho_corasick_matches():
    words = [("he", "HE"), ("she", "SHE"), ("hers", "HERS"), ("his", "HIS")]
    ours, theirs = compile_pair(words)
    for text in ("ushers", "ahishers", "she he", "nothing"):
        assert list(ours.iter(text)) == list(theirs.iter(text))


def test_failure_outputs_are_longest_to_shortest():
    words = [("a", "a"), ("aa", "aa"), ("aaa", "aaa")]
    ours, theirs = compile_pair(words)
    assert list(ours.iter("aaaa")) == list(theirs.iter("aaaa"))


def test_unicode_uses_python_character_indices():
    words = [("東京", 1), ("京", 2), ("café", 3), ("𐍈x", 4)]
    ours, theirs = compile_pair(words)
    text = "α東京 café 𐍈x"
    assert list(ours.iter(text)) == list(theirs.iter(text))


def test_dense_alphabet_lookup_falls_back_for_large_code_points():
    words = [("\U0010ffffx", 1), ("x", 2)]
    ours, theirs = compile_pair(words)
    text = "\U0010ffffx a"
    assert list(ours.iter(text)) == list(theirs.iter(text))


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, None), (2, 8), (2, 6), (-8, -2), (8, 2), (100, None)],
)
def test_slice_parity(start, end):
    ours, theirs = compile_pair([("he", 1), ("she", 2), ("hers", 3)])
    text = "xxushersyy"
    if end is None:
        assert list(ours.iter(text, start)) == list(theirs.iter(text, start))
    else:
        assert list(ours.iter(text, start, end)) == list(theirs.iter(text, start, end))


def test_ignore_ascii_whitespace():
    ours, theirs = compile_pair([("she", "she"), ("he", "he")])
    for text in ("s h e", "s\th\ne", "s\u00a0h\u00a0e"):
        assert list(ours.iter(text, ignore_white_space=True)) == list(
            theirs.iter(text, ignore_white_space=True)
        )


def test_iter_long_leftmost_longest():
    words = [("he", "HE"), ("she", "SHE"), ("hers", "HERS"), ("his", "HIS")]
    ours, theirs = compile_pair(words)
    for text in ("ushers", "ahishers", "she he", "hishershe"):
        assert list(ours.iter_long(text)) == list(theirs.iter_long(text))


def test_find_all_callback():
    ours, theirs = compile_pair([("ab", 1), ("b", 2), ("bc", 3)])
    got_ours = []
    got_theirs = []
    assert ours.find_all("zabc", lambda *match: got_ours.append(match)) is None
    assert theirs.find_all("zabc", lambda *match: got_theirs.append(match)) is None
    assert got_ours == got_theirs


def test_store_any_requires_and_replaces_values():
    ours = mojo.Automaton()
    theirs = upstream.Automaton()
    with pytest.raises(ValueError):
        ours.add_word("x")
    assert ours.add_word("x", object()) == theirs.add_word("x", object())
    assert ours.add_word("x", 7) == theirs.add_word("x", 7)
    assert ours.get("x") == theirs.get("x") == 7
    assert ours.kind == theirs.kind == mojo.TRIE


def test_store_ints_defaults_to_insertion_index():
    ours = mojo.Automaton(mojo.STORE_INTS)
    theirs = upstream.Automaton(upstream.STORE_INTS)
    for key in ("a", "b", "c"):
        assert ours.add_word(key) == theirs.add_word(key)
    assert [ours.get(key) for key in ("a", "b", "c")] == [
        theirs.get(key) for key in ("a", "b", "c")
    ]
    with pytest.raises(TypeError):
        ours.add_word("bad", "not-an-int")


def test_store_length():
    ours, theirs = paired(
        [("α", None), ("alphabet", None), ("東京", None)],
        store=mojo.STORE_LENGTH,
    )
    assert sorted(ours.items()) == sorted(theirs.items())
    with pytest.raises(ValueError):
        ours.add_word("x", 2)


def test_mapping_and_mutation_api():
    ours, theirs = paired([("alpha", 1), ("beta", 2), ("bet", 3)])
    assert len(ours) == len(theirs)
    assert ours.exists("beta") == theirs.exists("beta")
    assert ("missing" in ours) == ("missing" in theirs)
    assert ours.get("missing", 99) == theirs.get("missing", 99)
    assert ours.remove_word("beta") == theirs.remove_word("beta")
    assert ours.remove_word("beta") == theirs.remove_word("beta")
    assert ours.pop("bet") == theirs.pop("bet")
    with pytest.raises(KeyError):
        ours.pop("missing")
    ours.add_word("new", 4)
    theirs.add_word("new", 4)
    ours.remove_word("new")
    theirs.remove_word("new")
    assert set(ours.items()) == set(theirs.items())
    ours.clear()
    theirs.clear()
    assert (len(ours), ours.kind) == (len(theirs), theirs.kind)


@pytest.mark.parametrize("text", ["", "a", "alph", "alphabetic", "betamax", "z"])
def test_match_and_longest_prefix(text):
    ours, theirs = paired([("alphabet", 1), ("beta", 2)])
    assert ours.match(text) == theirs.match(text)
    assert ours.longest_prefix(text) == theirs.longest_prefix(text)


def test_key_value_item_filters():
    words = [(key, key.upper()) for key in ("foo", "food", "fob", "bar", "far", "fo")]
    ours, theirs = paired(words)
    cases = [
        (),
        ("fo",),
        ("fo?", "?"),
        ("fo?", "?", mojo.MATCH_EXACT_LENGTH),
        ("fo?", "?", mojo.MATCH_AT_LEAST_PREFIX),
        ("fo?", "?", mojo.MATCH_AT_MOST_PREFIX),
    ]
    for args in cases:
        assert sorted(ours.keys(*args)) == sorted(theirs.keys(*args))
        assert sorted(ours.values(*args)) == sorted(theirs.values(*args))
        assert sorted(ours.items(*args)) == sorted(theirs.items(*args))


def test_kind_transitions_and_search_guard():
    ours = mojo.Automaton()
    theirs = upstream.Automaton()
    assert ours.kind == theirs.kind == mojo.EMPTY
    with pytest.raises(AttributeError):
        list(ours.iter("text"))
    ours.make_automaton()
    theirs.make_automaton()
    assert ours.kind == theirs.kind == mojo.EMPTY
    ours.add_word("x", 1)
    theirs.add_word("x", 1)
    assert ours.kind == theirs.kind == mojo.TRIE
    ours.make_automaton()
    theirs.make_automaton()
    assert ours.kind == theirs.kind == mojo.AHOCORASICK
    assert ours.add_word("x", 2) == theirs.add_word("x", 2) is False
    assert ours.kind == theirs.kind == mojo.TRIE


def test_stats_describe_same_trie_shape():
    ours, theirs = paired([("he", 1), ("she", 2), ("hers", 3), ("his", 4)])
    a = ours.get_stats()
    b = theirs.get_stats()
    for key in ("nodes_count", "words_count", "longest_word", "links_count"):
        assert a[key] == b[key]
    assert set(a) == set(b)


def test_dump_has_complete_graph():
    automaton = mojo.Automaton()
    for index, word in enumerate(("he", "she", "hers", "his")):
        automaton.add_word(word, index)
    automaton.make_automaton()
    nodes, edges, failures = automaton.dump()
    stats = automaton.get_stats()
    assert len(nodes) == stats["nodes_count"]
    assert len(edges) == stats["links_count"]
    assert len(failures) == stats["nodes_count"] - 1
    assert sum(marker for _, marker in nodes) == len(automaton)


def test_sequence_keys_and_matching():
    words = [((1, 2), "12"), ((2,), "2"), ((1, 2, 3), "123")]
    ours, theirs = compile_pair(words, key_type=mojo.KEY_SEQUENCE)
    assert list(ours.iter((0, 1, 2, 3))) == list(theirs.iter((0, 1, 2, 3)))
    assert ours.exists((1, 2)) == theirs.exists((1, 2))
    with pytest.raises(TypeError):
        list(ours.iter([1, 2]))


def test_pickle_roundtrip():
    original = mojo.Automaton()
    original.add_word("he", {"id": 1})
    original.add_word("she", {"id": 2})
    original.make_automaton()
    restored = pickle.loads(pickle.dumps(original))
    assert restored.kind == mojo.AHOCORASICK
    assert list(restored.iter("she")) == list(original.iter("she"))


def test_save_load_roundtrip(tmp_path):
    original = mojo.Automaton()
    original.add_word("one", {"n": 1})
    original.add_word("two", {"n": 2})
    original.make_automaton()
    path = tmp_path / "automaton.bin"
    original.save(path, pickle.dumps)
    restored = mojo.load(path, pickle.loads)
    assert restored.kind == mojo.AHOCORASICK
    assert list(restored.iter("one two")) == list(original.iter("one two"))


def test_dense_output_reallocation_path():
    words = [("a" * length, length) for length in range(1, 65)]
    ours, theirs = compile_pair(words)
    text = "a" * 256
    assert list(ours.iter(text)) == list(theirs.iter(text))
    assert list(ours.iter(text * 2)) == list(theirs.iter(text * 2))


def test_simd_output_copy_with_scalar_tail():
    words = [("a" * length, length) for length in range(1, 10)]
    ours, theirs = compile_pair(words)
    assert list(ours.iter("a" * 20)) == list(theirs.iter("a" * 20))


@pytest.mark.parametrize("seed", range(10))
def test_randomized_match_parity(seed):
    rng = random.Random(seed)
    alphabet = "abcd"
    word_set = {
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9)))
        for _ in range(150)
    }
    words = [(word, word) for word in word_set]
    text = "".join(rng.choice(alphabet) for _ in range(3000))
    ours, theirs = compile_pair(words)
    assert list(ours.iter(text)) == list(theirs.iter(text))
    assert list(ours.iter_long(text)) == list(theirs.iter_long(text))
