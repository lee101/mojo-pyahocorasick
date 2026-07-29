# mojo-pyahocorasick

`mojo-pyahocorasick` is an Aho-Corasick multi-pattern matcher whose hot scan
loop is implemented in [Mojo](https://www.modular.com/mojo) and exposed through
a Python API modeled on
[`pyahocorasick`](https://github.com/WojciechMula/pyahocorasick).

It is intended for applications that build a dictionary of strings once and
scan many texts. The common `Automaton.add_word(...)`, `make_automaton()`, and
`iter(...)` workflow needs only an import change:

```python
import mojo_pyahocorasick as ahocorasick

automaton = ahocorasick.Automaton()
for word in ("he", "she", "hers", "his"):
    automaton.add_word(word, word)
automaton.make_automaton()

print(list(automaton.iter("ushers")))
# [(3, 'she'), (3, 'he'), (5, 'hers')]
```

## Coverage

The implemented subset is:

- `Automaton` with `KEY_STRING` and `KEY_SEQUENCE`;
- `STORE_ANY`, `STORE_INTS`, and `STORE_LENGTH`;
- `add_word`, `remove_word`, `pop`, `clear`, `exists`, `get`, `match`, and
  `longest_prefix`;
- `keys`, `values`, and `items`, including wildcard and match-mode filtering;
- `make_automaton`, `iter`, `iter_long`, and `find_all`;
- `kind`, `store`, `key_type`, `get_stats`, and `dump`;
- Python pickle support and paired `save` / `load` functions.

All constants use pyahocorasick's numeric values. Match order, inclusive
Unicode end indices, slice bounds, suffix outputs, whitespace handling, and
`iter_long`'s non-overlapping traversal are checked against the real
pyahocorasick 2.3.0 extension.

The following details are not covered:

- the upstream iterator objects' incremental `.set(...)` interface;
- binary compatibility with files written by pyahocorasick's native
  `save` format (this package's `save` and `load` round-trip their own format);
- upstream's lazy iterator and iterator-invalidation timing. This wrapper scans
  eagerly when `iter` is called and returns an iterator over the completed
  matches;
- Mojo acceleration for `iter_long`. Its unusual upstream-compatible selection
  logic currently runs in Python; standard `iter` and `find_all` use the Mojo
  kernel.

Trie construction and Python object ownership deliberately remain in Python.
The compute-heavy scan is the part ported to Mojo.

## Install

The repository pins its own Mojo nightly and Python environment:

```bash
pixi install
pixi run build
pixi run test
```

The supported build target is 64-bit Linux. A Mojo compiler and C-compatible
shared-library loader are required; this is currently a source-checkout
distribution, not a prebuilt wheel.

`pixi run build` emits
`dist/libmojo-pyahocorasick.so`. Importing the package also rebuilds the shared
library when it is absent or older than the Mojo source.

Run the example directly from the checkout with:

```bash
pixi run python - <<'PY'
import mojo_pyahocorasick as ahocorasick

a = ahocorasick.Automaton()
for word in ("he", "she", "hers", "his"):
    a.add_word(word, word)
a.make_automaton()
print(list(a.iter("ushers")))
PY
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, Mojo
1.0.0b3.dev2026072406, and pyahocorasick 2.3.0. Each row uses identical
patterns, text, and Python object results; the best of five warmed runs is
reported.

| workload | matches | mojo-pyahocorasick | pyahocorasick | result |
| --- | ---: | ---: | ---: | ---: |
| sparse ASCII: 1M chars, 1k patterns | 0 | 9.08 ms | 38.86 ms | 4.28x faster |
| DNA: 1M chars, 256 patterns | 3,944 | 5.74 ms | 27.03 ms | 4.71x faster |
| Unicode: 250k chars, 400 patterns | 21 | 1.54 ms | 7.31 ms | 4.76x faster |
| dense: 200k chars, 8 nested patterns | 1,599,972 | 603.01 ms | 340.87 ms | 1.77x slower |

The dense case emits almost 1.6 million Python tuples, so tuple and value
materialization is a substantial part of both timings. These are single-machine
best-case timings, not a general performance guarantee. There is no parallel or
GPU scan path.

## How it works

The Python layer builds failure links and two immutable transition layouts:

- sorted sparse edge arrays, valid for every automaton;
- a dense `nodes × observed alphabet` table when it contains at most 20 million
  entries.

The dense table stores 32-bit node indices. For bounded alphabets, a direct
code-point-to-column table turns each recognized input character into one
transition lookup and avoids both alphabet binary search and repeated
failure-link traversal. Large code points retain the sorted binary-search
lookup, and automata with a very large transition table automatically retain
the sparse edge path instead of consuming unbounded memory. Output ranges and
failure links are stored in flat 64-bit arrays; pattern IDs are 32-bit.

Output IDs and end positions are copied in native-width SIMD batches using
`simdwidthof[DType.float64]()` with a scalar remainder loop. Match-density
feedback sizes repeated output buffers up front, avoiding a count-and-rescan
cycle for dense workloads. Pattern values are gathered in NumPy before Python
tuples are produced.

Python strings are encoded as contiguous UTF-32 code points so returned indices
match Python character positions rather than UTF-8 byte offsets. NumPy owns
the text, automaton, and output buffers. A single `ctypes` call passes their
addresses as integers to `mpac_scan`; the `@export("mpac_scan") ... abi("C")`
wrapper rebuilds typed pointers inside Mojo. No Python object crosses the ABI.
Mojo returns pattern IDs and end indices, and Python maps those IDs back to the
original arbitrary values.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite contains 43 parity tests against pyahocorasick 2.3.0, plus four
FFI-boundary safety tests. It covers randomized automata, Unicode, integer
sequences, dense suffix output, store modes, SIMD remainder handling,
large-code-point fallback, filters, serialization, lifecycle behavior, and
buffer dtype/layout validation.

## License

MIT
