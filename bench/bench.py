"""Benchmark Mojo-backed matching against pyahocorasick on identical inputs."""

from __future__ import annotations

import os
import platform
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import ahocorasick as upstream  # noqa: E402
import mojo_pyahocorasick as mojo  # noqa: E402


def unique_words(count: int, length: int, alphabet: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < count:
        words.add("".join(rng.choice(alphabet) for _ in range(length)))
    return sorted(words)


def text(length: int, alphabet: str, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(length))


def automata(words: list[str]):
    ours = mojo.Automaton()
    theirs = upstream.Automaton()
    for index, word in enumerate(words):
        ours.add_word(word, index)
        theirs.add_word(word, index)
    ours.make_automaton()
    theirs.make_automaton()
    return ours, theirs


def best_time(function, repeat: int = 5) -> float:
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return min(samples)


def machine() -> str:
    cpu = platform.processor()
    if not cpu or cpu in {"x86_64", "AMD64", "aarch64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in handle
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = "unknown CPU"
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main() -> None:
    cases = []

    words = unique_words(1_000, 12, "abcdefghijklmnopqrstuvwxyz", 1)
    haystack = text(1_000_000, "abcdefghijklmnopqrstuvwxyz", 2)
    cases.append(("sparse ASCII: 1M chars, 1k patterns", words, haystack))

    words = unique_words(256, 8, "ACGT", 3)
    haystack = text(1_000_000, "ACGT", 4)
    cases.append(("DNA: 1M chars, 256 patterns", words, haystack))

    alphabet = "αβγδεζηθ東京大阪京都𐍈"
    words = unique_words(400, 6, alphabet, 5)
    haystack = text(250_000, alphabet, 6)
    cases.append(("Unicode: 250k chars, 400 patterns", words, haystack))

    words = ["a" * length for length in range(1, 9)]
    haystack = "a" * 200_000
    cases.append(("dense: 200k chars, 8 nested patterns", words, haystack))

    rows = []
    for name, patterns, haystack in cases:
        ours, theirs = automata(patterns)
        expected = list(theirs.iter(haystack))
        actual = list(ours.iter(haystack))
        if actual != expected:
            raise AssertionError(f"benchmark parity failed for {name}")
        ours_fn = lambda: list(ours.iter(haystack))
        theirs_fn = lambda: list(theirs.iter(haystack))
        ours_fn()
        theirs_fn()
        ours_time = best_time(ours_fn)
        theirs_time = best_time(theirs_fn)
        rows.append((name, ours_time, theirs_time, len(expected)))

    print(f"Machine: {machine()}")
    print()
    print("| workload | matches | mojo-pyahocorasick | pyahocorasick | result |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, ours_time, theirs_time, matches in rows:
        ratio = theirs_time / ours_time
        if ratio >= 1:
            result = f"{ratio:.2f}x faster"
        else:
            result = f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {matches:,} | {ours_time * 1e3:.2f} ms | "
            f"{theirs_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
