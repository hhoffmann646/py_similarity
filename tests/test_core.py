import math
import unittest

from py_similarity import levenshtein, jaro_winkler, levenshtein_ratio, jaro_similarity


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_empty_both(self):
        self.assertEqual(levenshtein("", ""), 0)

    def test_empty_one_side(self):
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_single_substitution(self):
        self.assertEqual(levenshtein("cat", "cot"), 1)

    def test_insertion(self):
        self.assertEqual(levenshtein("cat", "cats"), 1)

    def test_deletion(self):
        self.assertEqual(levenshtein("cats", "cat"), 1)

    def test_kitten_sitting(self):
        # Classic example from the literature: distance is 3.
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_order_independence(self):
        # Swapping arguments must not change the distance.
        self.assertEqual(
            levenshtein("saturday", "sunday"),
            levenshtein("sunday", "saturday"),
        )

    def test_type_error(self):
        with self.assertRaises(TypeError):
            levenshtein("abc", 123)  # type: ignore[arg-type]


class TestLevenshteinRatio(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(levenshtein_ratio("abc", "abc"), 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(levenshtein_ratio("", ""), 1.0)

    def test_one_empty(self):
        # ratio = 1 - len/len = 0.0
        self.assertAlmostEqual(levenshtein_ratio("abc", ""), 0.0)

    def test_partial(self):
        # "cat" -> "cot": distance 1, max_len 3, ratio = 1 - 1/3 = 2/3.
        self.assertAlmostEqual(levenshtein_ratio("cat", "cot"), 2.0 / 3.0)

    def test_range_is_unit_interval(self):
        # A handful of pairs; every result should be in [0, 1].
        for a, b in [("", ""), ("a", ""), ("", "a"), ("abc", "xyz"),
                     ("flaw", "lawn"), ("gumbo", "gambol")]:
            r = levenshtein_ratio(a, b)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


class TestJaro(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(jaro_similarity("abc", "abc"), 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(jaro_similarity("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(jaro_similarity("abc", ""), 0.0)
        self.assertAlmostEqual(jaro_similarity("", "abc"), 0.0)

    def test_no_matches(self):
        self.assertAlmostEqual(jaro_similarity("abc", "xyz"), 0.0)

    def test_single_char_inputs(self):
        # match_window is max(1//2 - 1, 0) = 0, so only same-index
        # characters can match.
        self.assertAlmostEqual(jaro_similarity("a", "a"), 1.0)
        self.assertAlmostEqual(jaro_similarity("a", "b"), 0.0)

    def test_transposition(self):
        # "CRATE" vs "TRACE" — a classic example where transposition
        # matters. Expected Jaro ≈ 0.7333333.
        self.assertAlmostEqual(
            jaro_similarity("CRATE", "TRACE"),
            0.7333333333333334,
            places=6,
        )

    def test_marth_vs_martha(self):
        # 5 matches (M, A, R, T, H), 1 transposition (T and H swapped).
        # Jaro = (5/6 + 5/5 + (5-1)/5) / 3 ≈ 0.8777778.
        self.assertAlmostEqual(
            jaro_similarity("MARTHA", "MARHT"),
            0.8777777777777779,
            places=6,
        )

    def test_type_error(self):
        with self.assertRaises(TypeError):
            jaro_similarity(123, "abc")  # type: ignore[arg-type]


class TestJaroWinkler(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(jaro_winkler("abc", "abc"), 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(jaro_winkler("", ""), 1.0)

    def test_no_common_prefix(self):
        # No shared prefix -> pure Jaro.
        a, b = "abc", "xyz"
        self.assertAlmostEqual(
            jaro_winkler(a, b),
            jaro_similarity(a, b),
        )

    def test_prefix_bonus_applied(self):
        # "MARTHA" vs "MARHT" has a 3-char common prefix.
        # jaro ≈ 0.8777778, bonus = 3 * 0.1 * (1 - jaro) ≈ 0.0366667.
        # Known value ≈ 0.9144444.
        self.assertAlmostEqual(
            jaro_winkler("MARTHA", "MARHT"),
            0.9144444444444445,
            places=6,
        )

    def test_prefix_cap_at_four(self):
        # A 5-char common prefix should still only boost by 4.
        long_prefix = "ABCDE" + "X"
        other = "ABCDE" + "Y"
        expected = jaro_similarity(long_prefix, other) + 4 * 0.1 * (
            1 - jaro_similarity(long_prefix, other)
        )
        self.assertAlmostEqual(
            jaro_winkler(long_prefix, other),
            expected,
            places=6,
        )

    def test_invalid_prefix_weight(self):
        with self.assertRaises(ValueError):
            jaro_winkler("abc", "abc", prefix_weight=0.3)
        with self.assertRaises(ValueError):
            jaro_winkler("abc", "abc", prefix_weight=-0.1)

    def test_type_error_on_non_string(self):
        with self.assertRaises(TypeError):
            jaro_winkler(["a"], "b")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
