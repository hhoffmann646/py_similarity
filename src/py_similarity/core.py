"""Core string similarity algorithms for fuzzy matching short strings.

All functions in this module operate on Python ``str`` objects only. They
were written for fuzzy-matching user input — autocomplete suggestions,
duplicate-detection, typo-tolerance — where the inputs are short and
readable rather than large or binary.

Design decisions:
  * The two-matrix-row Levenshtein keeps memory at O(min(m, n)) while
    preserving the exact edit distance. A full matrix would be clearer to
    read but would allocate m*n for every call, which is wasteful when the
    common use case is comparing 5-20 character tokens.
  * The Jaro transposition count is computed in a single pass after the
    matching window scan, so there is no second window scan.
  * The Jaro-Winkler prefix cap of 4 is the original Winkler (1990)
    formulation. We expose ``jaro_similarity`` separately so callers who
    do not want the common-prefix boost can still use it.
  * We do not expose an edit-distance-with-custom-costs API. If you need
    that, a different library is the right tool. Keeping the surface small
    keeps the code auditable.
"""

from __future__ import annotations

__all__ = [
    "levenshtein",
    "jaro_winkler",
    "levenshtein_ratio",
    "jaro_similarity",
]


def levenshtein(s1: str, s2: str) -> int:
    """Return the Levenshtein edit distance between ``s1`` and ``s2``.

    The distance is the minimum number of single-character insertions,
    deletions, and substitutions needed to turn ``s1`` into ``s2``.

    Edge cases:
      * If both strings are empty, returns ``0``.
      * If exactly one string is empty, returns the length of the other.
      * Accepts any ``str``, including strings containing surrogates or
        combining marks. Character indices are used as-is; no
        normalisation is performed.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("both arguments must be str")

    if s1 == s2:
        return 0

    if len(s1) < len(s2):
        # Ensure s1 is the longer string so we allocate the shorter row.
        s1, s2 = s2, s1

    len2 = len(s2)
    if len2 == 0:
        return len(s1)

    # Previous row of distances (for prefixes of s2 of length 0..len2).
    # Using a plain list rather than array.array keeps the inner loop
    # pure-Python and avoids a C extension call per element, which is
    # measurably faster on CPython for the small sizes this library
    # targets.
    previous_row = list(range(len2 + 1))

    for i, c1 in enumerate(s1, start=1):
        # The substitution cost at the start of this row is i (deleting
        # all of s1[:i] to reach the empty prefix of s2).
        current_row = [i]
        for j, c2 in enumerate(s2, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            if c1 == c2:
                sub_cost = previous_row[j - 1]
            else:
                sub_cost = previous_row[j - 1] + 1
            current_row.append(min(insert_cost, delete_cost, sub_cost))
        previous_row = current_row

    return previous_row[len2]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Return a normalised Levenshtein similarity in ``[0.0, 1.0]``.

    ``ratio = 1 - distance / max(len(s1), len(s2))``.

    This is the ratio used throughout the library because it stays in a
    fixed range and is therefore easy to threshold. Two identical strings
    yield ``1.0``; two strings with no characters in common yield ``0.0``
    only when at least one is non-empty and both have the same length —
    otherwise the floor is above zero because some insertions are needed.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("both arguments must be str")

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein(s1, s2) / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """Return the Jaro similarity between ``s1`` and ``s2``.

    Returns a float in ``[0.0, 1.0]``. ``1.0`` means the strings are
    identical; ``0.0`` means no characters match within the window.

    The matching window is ``floor(max(|s1|, |s2|) / 2) - 1``, clamped to
    be at least 0. This is the standard Jaro (1989) definition.

    Edge cases:
      * Both empty -> ``1.0`` (they are trivially identical).
      * Exactly one empty -> ``0.0`` (no matches possible).
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("both arguments must be str")

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0

    # Window of allowed match distance. The ``- 1`` is part of Jaro's
    # original definition; the ``max(..., 0)`` guards against negative
    # windows for single-character inputs, where the formula would
    # otherwise yield -1.
    match_window = max(max(len1, len2) // 2 - 1, 0)

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        # Scan s2 within [i - window, i + window], clamped to its bounds.
        # This single forward scan is O(len1 * (2*window + 1)), which is
        # fine for short strings and avoids a separate pass.
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions: walk the matched characters of s1 and s2 in
    # order; each position where the characters differ is one half
    # transposition.
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    m = float(matches)
    return (m / len1 + m / len2 + (m - transpositions) / m) / 3.0


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Return the Jaro-Winkler similarity between ``s1`` and ``s2``.

    This is Jaro similarity plus a bonus for sharing a common prefix up to
    4 characters long. The bonus is ``prefix_length * prefix_weight * (1 -
    jaro)``, so identical strings (Jaro = 1.0) receive no bonus and the
    result stays in ``[0.0, 1.0]``.

    ``prefix_weight`` defaults to ``0.1`` — the value Winkler (1990)
    used. Values above ``0.25`` are rejected because they can push the
    score above 1.0 for long common prefixes, which would break the
    ``[0, 1]`` contract.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("both arguments must be str")
    if not isinstance(prefix_weight, (int, float)):
        raise TypeError("prefix_weight must be a number")
    if prefix_weight < 0 or prefix_weight > 0.25:
        raise ValueError("prefix_weight must be in [0, 0.25]")

    jaro = jaro_similarity(s1, s2)

    prefix_len = 0
    for a, b in zip(s1, s2):
        if a != b or prefix_len == 4:
            break
        prefix_len += 1

    return jaro + prefix_len * prefix_weight * (1.0 - jaro)
