"""Basic synthetic invariants for the provisional best-quote reconstruction.

No licensed market data is needed. Run: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lobster2"))
from reconstruct_ofi import Book, ofi


class BookTests(unittest.TestCase):
    def test_order_lifecycle_and_quote(self):
        book = Book()
        self.assertTrue(book.update(1, 100, 50, 1000, 1))
        self.assertTrue(book.update(1, 200, 90, 1002, -1))
        self.assertEqual(book.quote(), ((1000, 50), (1002, 90)))
        self.assertTrue(book.update(4, 100, 20, 1000, 1))
        self.assertEqual(book.quote()[0], (1000, 30))
        self.assertTrue(book.update(3, 100, 30, 1000, 1))
        self.assertIsNone(book.quote())

    def test_ofi_signs(self):
        original = ((1000, 50), (1002, 70))
        more_bid = ((1000, 75), (1002, 70))
        more_ask = ((1000, 50), (1002, 90))
        self.assertEqual(ofi(original, more_bid), 25)
        self.assertEqual(ofi(original, more_ask), -20)

    def test_hidden_executions_do_not_modify_displayed_depth(self):
        book = Book()
        book.update(1, 1, 50, 1000, 1)
        book.update(1, 2, 80, 1001, -1)
        self.assertFalse(book.update(5, 99, 30, 1001, -1))
        self.assertEqual(book.quote(), ((1000, 50), (1001, 80)))


if __name__ == "__main__":
    unittest.main()
