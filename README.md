# py_similarity

A small, zero-dependency Python library providing Levenshtein edit distance and Jaro-Winkler similarity for fuzzy matching short strings.

## Usage

```python
from py_similarity import levenshtein, jaro_winkler, levenshtein_ratio, jaro_similarity

levenshtein("kitten", "sitting")        # -> 3
levenshtein_ratio("cat", "cot")          # -> 0.6666...
jaro_similarity("MARTHA", "MARHT")       # -> 0.9444...
jaro_winkler("MARTHA", "MARHT")          # -> 0.9666...
```

## Why this exists

This library was written for the specific case of fuzzy-matching short user input — autocomplete suggestions, duplicate detection, typo tolerance — where the inputs are readable strings of a few dozen characters at most. The standard-library-only constraint means it can be dropped into a locked-down environment with no package index access. The trade-off is that the algorithms are written in pure Python and are not suitable for bulk comparison of very long strings or large corpora; a C-backed library will be faster there.

## Edge cases worth knowing

- Both empty strings return similarity `1.0` and distance `0`. Exactly one empty string returns `0.0` similarity and the length of the non-empty string as distance.
- `jaro_winkler` accepts a `prefix_weight` keyword (default `0.1`, Winkler 1990) but rejects values above `0.25`, because longer common prefixes would otherwise push the result above `1.0`.
- The Jaro matching window is `max(len // 2 - 1, 0)`. Single-character strings thus only match at the same index, which is the standard definition but worth noting if you are comparing very short tokens.
- No Unicode normalisation is performed. Strings are compared by code unit index. Pass normalised input if your use case requires it.

## Exported names

`levenshtein(s1: str, s2: str) -> int`
`levenshtein_ratio(s1: str, s2: str) -> float`
`jaro_similarity(s1: str, s2: str) -> float`
`jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float`
