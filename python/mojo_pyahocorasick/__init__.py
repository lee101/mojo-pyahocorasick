"""A Python-compatible Aho-Corasick automaton with its scan loop in Mojo."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle
from typing import Any, Callable, Iterator

import numpy as np

from ._lib import scan as _mojo_scan

EMPTY = 0
TRIE = 1
AHOCORASICK = 2

STORE_INTS = 10
STORE_LENGTH = 20
STORE_ANY = 30

KEY_STRING = 100
KEY_SEQUENCE = 200

MATCH_EXACT_LENGTH = 0
MATCH_AT_MOST_PREFIX = 1
MATCH_AT_LEAST_PREFIX = 2

unicode = 1
__version__ = "0.1.0"

_MISSING = object()
_FILE_MAGIC = b"MOJO-PYAHOCORASICK\x00\x01"
_MAX_MATCH_PREALLOCATION = 4_000_000
_I32_MAX = np.iinfo(np.int32).max


def _address(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if not address:
        raise RuntimeError("NumPy returned a null buffer address")
    return address


def _nonempty(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if array.size:
        return np.ascontiguousarray(array, dtype=dtype)
    return np.zeros(1, dtype=dtype)


def _ffi_array(
    name: str, array: np.ndarray | None, dtype: np.dtype
) -> np.ndarray:
    """Validate buffers before their addresses cross the C ABI."""
    if not isinstance(array, np.ndarray):
        raise RuntimeError(f"invalid {name} buffer")
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise RuntimeError(
            f"invalid {name} dtype: expected {expected}, got {array.dtype}"
        )
    if array.ndim != 1 or not array.flags.c_contiguous or not array.flags.aligned:
        raise RuntimeError(f"invalid {name} layout")
    return array


class Automaton:
    """Automaton(value_type=STORE_ANY, key_type=KEY_STRING)."""

    def __init__(self, value_type: int = STORE_ANY, key_type: int = KEY_STRING):
        if value_type not in (STORE_ANY, STORE_INTS, STORE_LENGTH):
            raise ValueError(
                "store value must be one of STORE_LENGTH, STORE_INTS or STORE_ANY"
            )
        if key_type not in (KEY_STRING, KEY_SEQUENCE):
            raise ValueError("key_type must have value KEY_STRING or KEY_SEQUENCE")
        self._store = value_type
        self._key_type = key_type
        self._kind = EMPTY
        self._words: dict[str | tuple[int, ...], Any] = {}
        self._clear_compiled()

    @property
    def kind(self) -> int:
        return self._kind

    @property
    def store(self) -> int:
        return self._store

    @property
    def key_type(self) -> int:
        return self._key_type

    def _clear_compiled(self) -> None:
        self._edge_starts: np.ndarray | None = None
        self._edge_labels: np.ndarray | None = None
        self._edge_targets: np.ndarray | None = None
        self._failure: np.ndarray | None = None
        self._output_starts: np.ndarray | None = None
        self._output_ids: np.ndarray | None = None
        self._pattern_keys: tuple[str | tuple[int, ...], ...] = ()
        self._pattern_values: tuple[Any, ...] = ()
        self._pattern_value_array: np.ndarray | None = None
        self._pattern_lengths: np.ndarray | None = None
        self._children: tuple[dict[int, int], ...] = ()
        self._terminal_ids: tuple[int, ...] = ()
        self._alphabet: np.ndarray | None = None
        self._alphabet_lookup: np.ndarray | None = None
        self._dense_transitions: np.ndarray | None = None
        self._max_outputs = 0
        self._matches_per_unit = 0.0

    def _key(self, key: Any) -> str | tuple[int, ...]:
        if self._key_type == KEY_STRING:
            if not isinstance(key, str):
                raise TypeError("string expected")
            return key
        if not isinstance(key, tuple):
            raise TypeError("argument is not a supported sequence type")
        result = []
        for item in key:
            if not isinstance(item, int):
                raise TypeError("item #0 is not a number")
            if item < -(1 << 31) or item >= (1 << 32):
                raise ValueError("item value outside range")
            result.append(int(item) & 0xFFFFFFFF)
        return tuple(result)

    @staticmethod
    def _key_units(key: str | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(key, str):
            return tuple(map(ord, key))
        return key

    def _text_units(self, text: Any) -> np.ndarray:
        if self._key_type == KEY_STRING:
            if not isinstance(text, str):
                raise TypeError("string expected")
            if not text:
                return np.zeros(0, dtype=np.uint32)
            return np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
        if not isinstance(text, tuple):
            raise TypeError("tuple required")
        values = self._key(text)
        return np.asarray(values, dtype=np.uint32)

    def add_word(self, key: Any, value: Any = _MISSING) -> bool:
        normalized = self._key(key)
        if not normalized:
            return False
        if self._store == STORE_ANY:
            if value is _MISSING:
                raise ValueError("A value object is required as second argument.")
            stored = value
        elif self._store == STORE_LENGTH:
            if value is not _MISSING:
                raise ValueError("STORE_LENGTH does not accept a value")
            stored = len(normalized)
        else:
            stored = len(self._words) + 1 if value is _MISSING else value
            if not isinstance(stored, int):
                raise TypeError("an integer is required")
            if stored < -(1 << 31) or stored >= (1 << 31):
                raise OverflowError("signed integer is greater than maximum")
            stored = int(stored)

        inserted = normalized not in self._words
        self._words[normalized] = stored
        self._kind = TRIE
        self._clear_compiled()
        return inserted

    def make_automaton(self) -> None:
        if not self._words:
            self._kind = EMPTY
            self._clear_compiled()
            return None

        children: list[dict[int, int]] = [{}]
        terminal: list[int] = [-1]
        pattern_keys = tuple(self._words)
        if len(pattern_keys) > _I32_MAX:
            raise OverflowError("too many patterns for the 32-bit Mojo ABI")
        for pattern_id, key in enumerate(pattern_keys):
            state = 0
            for label in self._key_units(key):
                child = children[state].get(label)
                if child is None:
                    child = len(children)
                    if child > _I32_MAX:
                        raise OverflowError("too many trie nodes for the 32-bit dense ABI")
                    children[state][label] = child
                    children.append({})
                    terminal.append(-1)
                state = child
            terminal[state] = pattern_id

        failure = [0] * len(children)
        outputs: list[list[int]] = [[] for _ in children]
        queue: deque[int] = deque()
        breadth_first = [0]
        for child in children[0].values():
            queue.append(child)
        while queue:
            state = queue.popleft()
            breadth_first.append(state)
            if terminal[state] >= 0:
                outputs[state].append(terminal[state])
            if failure[state]:
                outputs[state].extend(outputs[failure[state]])
            for label, child in children[state].items():
                fallback = failure[state]
                while fallback and label not in children[fallback]:
                    fallback = failure[fallback]
                candidate = children[fallback].get(label)
                if candidate is not None and candidate != child:
                    failure[child] = candidate
                queue.append(child)

        edge_starts = [0]
        edge_labels: list[int] = []
        edge_targets: list[int] = []
        for edges in children:
            for label, target in sorted(edges.items()):
                edge_labels.append(label)
                edge_targets.append(target)
            edge_starts.append(len(edge_labels))

        output_starts = [0]
        output_ids: list[int] = []
        for node_outputs in outputs:
            output_ids.extend(node_outputs)
            output_starts.append(len(output_ids))

        self._edge_starts = np.asarray(edge_starts, dtype=np.int64)
        self._edge_labels = _nonempty(
            np.asarray(edge_labels, dtype=np.uint32), np.dtype(np.uint32)
        )
        self._edge_targets = _nonempty(
            np.asarray(edge_targets, dtype=np.int64), np.dtype(np.int64)
        )
        self._failure = np.asarray(failure, dtype=np.int64)
        self._output_starts = np.asarray(output_starts, dtype=np.int64)
        self._output_ids = _nonempty(
            np.asarray(output_ids, dtype=np.int32), np.dtype(np.int32)
        )
        self._max_outputs = max(map(len, outputs))
        self._pattern_keys = pattern_keys
        self._pattern_values = tuple(self._words[key] for key in pattern_keys)
        self._pattern_value_array = np.empty(len(pattern_keys), dtype=object)
        self._pattern_value_array[:] = self._pattern_values
        self._pattern_lengths = np.asarray(
            [len(key) for key in pattern_keys], dtype=np.int64
        )
        self._children = tuple(children)
        self._terminal_ids = tuple(terminal)
        alphabet_values = sorted({label for edges in children for label in edges})
        if len(children) * len(alphabet_values) <= 20_000_000:
            alphabet_index = {
                label: index for index, label in enumerate(alphabet_values)
            }
            dense = np.zeros(
                (len(children), len(alphabet_values)), dtype=np.int32
            )
            for state in breadth_first:
                if state:
                    dense[state] = dense[failure[state]]
                for label, target in children[state].items():
                    dense[state, alphabet_index[label]] = target
            self._alphabet = _nonempty(
                np.asarray(alphabet_values, dtype=np.uint32), np.dtype(np.uint32)
            )
            if alphabet_values and alphabet_values[-1] < 1_048_576:
                self._alphabet_lookup = np.full(
                    alphabet_values[-1] + 1, -1, dtype=np.int32
                )
                self._alphabet_lookup[self._alphabet] = np.arange(
                    len(alphabet_values), dtype=np.int32
                )
            else:
                self._alphabet_lookup = np.zeros(1, dtype=np.int32)
            self._dense_transitions = _nonempty(
                dense.reshape(-1), np.dtype(np.int32)
            )
        else:
            self._alphabet = np.zeros(1, dtype=np.uint32)
            self._alphabet_lookup = np.zeros(1, dtype=np.int32)
            self._dense_transitions = np.zeros(1, dtype=np.int32)
        self._kind = AHOCORASICK
        return None

    def _require_automaton(self) -> None:
        if self._kind != AHOCORASICK:
            raise AttributeError(
                "Not an Aho-Corasick automaton yet: call make_automaton() first"
            )

    def _search_ids(
        self,
        text: Any,
        start: int = 0,
        end: int | None = None,
        ignore_white_space: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_automaton()
        units = self._text_units(text)
        text_length = len(units)
        first = min(max(int(start), 0), text_length)
        last = text_length if end is None else min(max(int(end), 0), text_length)
        if last < first:
            last = first
        selected = units[first:last]
        capacity = max(16, min(len(selected), 65_536))
        if selected.size and self._matches_per_unit:
            predicted = min(
                int(len(selected) * self._matches_per_unit) + 64,
                len(selected) * self._max_outputs,
                _MAX_MATCH_PREALLOCATION,
            )
            capacity = max(capacity, predicted)
        end_indices = np.empty(capacity, dtype=np.int64)
        match_ids = np.empty(capacity, dtype=np.int32)

        arrays = (
            _ffi_array("edge starts", self._edge_starts, np.int64),
            _ffi_array("edge labels", self._edge_labels, np.uint32),
            _ffi_array("edge targets", self._edge_targets, np.int64),
            _ffi_array("failure links", self._failure, np.int64),
            _ffi_array("output starts", self._output_starts, np.int64),
            _ffi_array("output IDs", self._output_ids, np.int32),
        )
        alphabet = _ffi_array("alphabet", self._alphabet, np.uint32)
        alphabet_lookup = _ffi_array(
            "alphabet lookup", self._alphabet_lookup, np.int32
        )
        dense_transitions = _ffi_array(
            "dense transitions", self._dense_transitions, np.int32
        )
        selected = _ffi_array("text", selected, np.uint32)
        end_indices = _ffi_array("end indices", end_indices, np.int64)
        match_ids = _ffi_array("match IDs", match_ids, np.int32)

        node_count = len(self._children)
        if (
            len(arrays[0]) != node_count + 1
            or len(arrays[3]) != node_count
            or len(arrays[4]) != node_count + 1
        ):
            raise RuntimeError("inconsistent automaton buffer lengths")
        alphabet_size = (
            len(alphabet)
            if dense_transitions.size > 1 or node_count == 1
            else 0
        )
        alphabet_lookup_size = (
            len(alphabet_lookup) if alphabet_lookup.size > 1 else 0
        )
        alphabet_index = (
            alphabet_lookup if alphabet_lookup_size else alphabet
        )
        if alphabet_size and dense_transitions.size != node_count * alphabet_size:
            raise RuntimeError("inconsistent dense transition buffer length")

        count = _mojo_scan(
            _address(selected),
            len(selected),
            *(_address(array) for array in arrays),
            _address(alphabet_index),
            _address(dense_transitions),
            alphabet_size,
            alphabet_lookup_size,
            _address(end_indices),
            _address(match_ids),
            capacity,
            int(bool(ignore_white_space)),
        )
        if count > capacity:
            capacity = count
            end_indices = np.empty(capacity, dtype=np.int64)
            match_ids = np.empty(capacity, dtype=np.int32)
            second_count = _mojo_scan(
                _address(selected),
                len(selected),
                *(_address(array) for array in arrays),
                _address(alphabet_index),
                _address(self._dense_transitions),
                alphabet_size,
                alphabet_lookup_size,
                _address(end_indices),
                _address(match_ids),
                capacity,
                int(bool(ignore_white_space)),
            )
            if second_count != count:
                raise RuntimeError("automaton changed during iteration")
        if selected.size:
            self._matches_per_unit = count / len(selected)
        end_indices = end_indices[:count]
        end_indices += first
        return end_indices, match_ids[:count]

    def iter(
        self,
        string: Any,
        start: int = 0,
        end: int | None = None,
        ignore_white_space: bool = False,
    ) -> Iterator[tuple[int, Any]]:
        ends, ids = self._search_ids(string, start, end, ignore_white_space)
        assert self._pattern_value_array is not None
        values = self._pattern_value_array[ids].tolist()
        return zip(ends.tolist(), values)

    def iter_long(
        self, string: Any, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, Any]]:
        self._require_automaton()
        units = self._text_units(string)
        text_length = len(units)
        first = min(max(int(start), 0), text_length)
        last = text_length if end is None else min(max(int(end), 0), text_length)
        if last < first:
            last = first
        children = self._children
        terminal = self._terminal_ids
        assert self._failure is not None

        selected: list[tuple[int, Any]] = []
        state = 0
        index = first
        last_index = -1
        last_pattern = -1
        fail_node = -1
        fail_index = -1
        fail_flag = False
        while index < last or last_pattern >= 0:
            if index >= last:
                key = self._pattern_keys[last_pattern]
                selected.append((last_index, self._words[key]))
                state = 0
                index = last_index + 1
                last_pattern = -1
                last_index = -1
                fail_node = -1
                fail_index = -1
                fail_flag = False
                continue
            label = int(units[index])
            if fail_flag:
                next_state = int(self._failure[fail_node])
                index = fail_index
                fail_node = -1
                fail_index = -1
                fail_flag = False
            else:
                next_state = children[state].get(label, -1)

            if next_state >= 0:
                pattern_id = terminal[next_state]
                if pattern_id >= 0:
                    last_pattern = pattern_id
                    last_index = index
                elif (
                    last_pattern < 0
                    and fail_node < 0
                    and int(self._failure[next_state]) != 0
                    and terminal[int(self._failure[next_state])] >= 0
                ):
                    fail_node = next_state
                    fail_index = index
                state = next_state
                index += 1
                continue

            if last_pattern >= 0:
                key = self._pattern_keys[last_pattern]
                selected.append((last_index, self._words[key]))
                state = 0
                index = last_index + 1
                last_pattern = -1
                last_index = -1
                fail_node = -1
                fail_index = -1
                fail_flag = False
                continue
            if fail_node >= 0:
                fail_flag = True
                continue

            while True:
                if state == 0:
                    index += 1
                    break
                state = int(self._failure[state])
                if label in children[state]:
                    break

        return iter(selected)

    def find_all(
        self,
        string: Any,
        callback: Callable[[int, Any], Any],
        start: int = 0,
        end: int | None = None,
    ) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        for end_index, value in self.iter(string, start, end):
            callback(end_index, value)
        return None

    def exists(self, key: Any) -> bool:
        return self._key(key) in self._words

    def match(self, key: Any) -> bool:
        normalized = self._key(key)
        return any(word[: len(normalized)] == normalized for word in self._words)

    def longest_prefix(self, string: Any) -> int:
        normalized = self._key(string)
        best = 0
        for word in self._words:
            limit = min(len(word), len(normalized))
            index = 0
            while index < limit and word[index] == normalized[index]:
                index += 1
            best = max(best, index)
        return best

    def get(self, key: Any, default: Any = _MISSING) -> Any:
        normalized = self._key(key)
        if normalized in self._words:
            return self._words[normalized]
        if default is not _MISSING:
            return default
        raise KeyError

    def remove_word(self, word: Any) -> bool:
        normalized = self._key(word)
        if normalized not in self._words:
            return False
        del self._words[normalized]
        self._kind = TRIE if self._words else EMPTY
        self._clear_compiled()
        return True

    def pop(self, word: Any) -> Any:
        normalized = self._key(word)
        if normalized not in self._words:
            raise KeyError
        value = self._words.pop(normalized)
        self._kind = TRIE if self._words else EMPTY
        self._clear_compiled()
        return value

    def clear(self) -> None:
        self._words.clear()
        self._kind = EMPTY
        self._clear_compiled()

    def _filtered_keys(
        self,
        prefix: Any,
        wildcard: Any,
        how: int,
    ) -> Iterator[str | tuple[int, ...]]:
        if prefix is _MISSING:
            yield from self._words
            return
        normalized = self._key(prefix)
        if wildcard is _MISSING:
            for key in self._words:
                if key[: len(normalized)] == normalized:
                    yield key
            return
        wildcard_key = self._key(wildcard)
        if len(wildcard_key) != 1:
            raise ValueError("wildcard must be a single character")
        if how not in (
            MATCH_EXACT_LENGTH,
            MATCH_AT_MOST_PREFIX,
            MATCH_AT_LEAST_PREFIX,
        ):
            raise ValueError("invalid match mode")
        wildcard_unit = wildcard_key[0]
        for key in self._words:
            if how == MATCH_EXACT_LENGTH and len(key) != len(normalized):
                continue
            if how == MATCH_AT_LEAST_PREFIX and len(key) < len(normalized):
                continue
            if how == MATCH_AT_MOST_PREFIX and len(key) > len(normalized):
                continue
            common = min(len(key), len(normalized))
            if all(
                normalized[index] == wildcard_unit
                or normalized[index] == key[index]
                for index in range(common)
            ):
                yield key

    def keys(
        self,
        prefix: Any = _MISSING,
        wildcard: Any = _MISSING,
        how: int = MATCH_EXACT_LENGTH,
    ) -> Iterator[str | tuple[int, ...]]:
        return self._filtered_keys(prefix, wildcard, how)

    def values(
        self,
        prefix: Any = _MISSING,
        wildcard: Any = _MISSING,
        how: int = MATCH_EXACT_LENGTH,
    ) -> Iterator[Any]:
        return (
            self._words[key] for key in self._filtered_keys(prefix, wildcard, how)
        )

    def items(
        self,
        prefix: Any = _MISSING,
        wildcard: Any = _MISSING,
        how: int = MATCH_EXACT_LENGTH,
    ) -> Iterator[tuple[str | tuple[int, ...], Any]]:
        return (
            (key, self._words[key])
            for key in self._filtered_keys(prefix, wildcard, how)
        )

    def get_stats(self) -> dict[str, int]:
        children: list[dict[int, int]] = [{}]
        for key in self._words:
            state = 0
            for label in self._key_units(key):
                if label not in children[state]:
                    children[state][label] = len(children)
                    children.append({})
                state = children[state][label]
        nodes = len(children)
        links = nodes - 1
        sizeof_node = 32
        return {
            "nodes_count": nodes,
            "words_count": len(self._words),
            "longest_word": max((len(key) for key in self._words), default=0),
            "links_count": links,
            "sizeof_node": sizeof_node,
            "total_size": nodes * sizeof_node + links * 8,
        }

    def dump(self) -> tuple[list[tuple[int, int]], list[tuple[int, Any, int]], list[tuple[int, int]]]:
        children: list[dict[int, int]] = [{}]
        terminal: set[int] = set()
        for key in self._words:
            state = 0
            for label in self._key_units(key):
                if label not in children[state]:
                    children[state][label] = len(children)
                    children.append({})
                state = children[state][label]
            terminal.add(state)
        nodes = [(index, int(index in terminal)) for index in range(len(children))]
        edges = [
            (source, label, target)
            for source, child_map in enumerate(children)
            for label, target in child_map.items()
        ]
        failures: list[tuple[int, int]] = []
        if self._kind == AHOCORASICK and self._failure is not None:
            failures = [
                (index, int(self._failure[index]))
                for index in range(1, len(self._failure))
            ]
        return nodes, edges, failures

    def save(self, path: str | Path, serializer: Callable[[Any], bytes]) -> None:
        if self._store == STORE_ANY:
            if not callable(serializer):
                raise TypeError("serializer must be callable")
            entries = [(key, serializer(value)) for key, value in self._words.items()]
        else:
            entries = list(self._words.items())
        payload = (self._store, self._key_type, self._kind, entries)
        with open(path, "wb") as handle:
            handle.write(_FILE_MAGIC)
            pickle.dump(payload, handle, protocol=5)

    def __len__(self) -> int:
        return len(self._words)

    def __iter__(self) -> Iterator[str | tuple[int, ...]]:
        return self.keys()

    def __contains__(self, key: Any) -> bool:
        return self.exists(key)

    def __reduce__(self):
        return (
            Automaton,
            (self._store, self._key_type),
            (self._kind, list(self._words.items())),
        )

    def __setstate__(self, state: tuple[int, list[tuple[Any, Any]]]) -> None:
        kind, entries = state
        self._words = dict(entries)
        self._kind = TRIE if entries else EMPTY
        self._clear_compiled()
        if kind == AHOCORASICK:
            self.make_automaton()


def load(path: str | Path, deserializer: Callable[[bytes], Any]) -> Automaton:
    with open(path, "rb") as handle:
        if handle.read(len(_FILE_MAGIC)) != _FILE_MAGIC:
            raise ValueError("not a mojo-pyahocorasick automaton file")
        store, key_type, kind, entries = pickle.load(handle)
    automaton = Automaton(store, key_type)
    if store == STORE_ANY:
        if not callable(deserializer):
            raise TypeError("deserializer must be callable")
        automaton._words = {key: deserializer(value) for key, value in entries}
    else:
        automaton._words = dict(entries)
    automaton._kind = TRIE if entries else EMPTY
    if kind == AHOCORASICK:
        automaton.make_automaton()
    return automaton


__all__ = [
    "AHOCORASICK",
    "Automaton",
    "EMPTY",
    "KEY_SEQUENCE",
    "KEY_STRING",
    "MATCH_AT_LEAST_PREFIX",
    "MATCH_AT_MOST_PREFIX",
    "MATCH_EXACT_LENGTH",
    "STORE_ANY",
    "STORE_INTS",
    "STORE_LENGTH",
    "TRIE",
    "load",
    "unicode",
]
