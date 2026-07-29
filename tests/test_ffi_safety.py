from __future__ import annotations

import numpy as np
import pytest

import mojo_pyahocorasick as mojo
from mojo_pyahocorasick import _lib


def compiled() -> mojo.Automaton:
    automaton = mojo.Automaton()
    automaton.add_word("needle", 1)
    automaton.make_automaton()
    return automaton


def test_binding_rejects_wrong_argument_count_before_call():
    with pytest.raises(TypeError, match="expects 16 arguments"):
        _lib.scan(1, 2)


def test_ffi_rejects_wrong_dtype_before_call():
    automaton = compiled()
    automaton._failure = automaton._failure.astype(np.int32)
    with pytest.raises(RuntimeError, match="failure links dtype"):
        list(automaton.iter("needle"))


def test_ffi_rejects_noncontiguous_buffer_before_call():
    automaton = compiled()
    original = automaton._edge_starts
    storage = np.empty(original.size * 2, dtype=np.int64)
    storage[::2] = original
    automaton._edge_starts = storage[::2]
    with pytest.raises(RuntimeError, match="edge starts layout"):
        list(automaton.iter("needle"))


def test_ffi_rejects_inconsistent_lengths_before_call():
    automaton = compiled()
    automaton._output_starts = automaton._output_starts[:-1]
    with pytest.raises(RuntimeError, match="inconsistent automaton buffer lengths"):
        list(automaton.iter("needle"))
