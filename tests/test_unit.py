"""
Unit tests for the document splitter.
Tests basic functionality, integration with real documents, and edge cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.document_splitter import split_document_into_sections
from src.document_splitter.adjustments import combine_sections
from src.document_splitter.fallbacks import fallback_split


class TestBasicFunctionality(unittest.TestCase):
    """Test individual helper functions"""

    def test_combine_sections(self):
        sections = ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5"]
        result = combine_sections(sections, 3)
        self.assertEqual(len(result), 3)

    def test_fallback_split(self):
        # Fallback should split on paragraph boundaries
        document = "Paragraph 1\n\nParagraph 2\n\nParagraph 3\n\nParagraph 4"
        result = fallback_split(document, 2, combine_sections)
        self.assertEqual(len(result), 2)


class TestIntegration(unittest.TestCase):
    """Test with a real document from examples/"""

    @classmethod
    def setUpClass(cls):
        doc_path = Path(__file__).parent.parent / "examples" / "sample_document.md"
        if doc_path.exists():
            cls.sample_doc = doc_path.read_text()
        else:
            cls.sample_doc = None

    def setUp(self):
        if not self.sample_doc:
            self.skipTest("sample_document.md not found")

    def test_split_12_sections(self):
        sections = split_document_into_sections(self.sample_doc, 12)
        self.assertEqual(len(sections), 12)

        # Make sure no content was lost or changed
        combined = "\n\n".join(sections)
        original_normalized = " ".join(self.sample_doc.split())
        combined_normalized = " ".join(combined.split())
        self.assertEqual(original_normalized, combined_normalized)

    def test_split_37_sections(self):
        sections = split_document_into_sections(self.sample_doc, 37)
        self.assertEqual(len(sections), 37)

        # Make sure no content was lost or changed
        combined = "\n\n".join(sections)
        original_normalized = " ".join(self.sample_doc.split())
        combined_normalized = " ".join(combined.split())
        self.assertEqual(original_normalized, combined_normalized)


class TestEdgeCases(unittest.TestCase):
    """Test boundary conditions and error handling"""

    def test_single_section(self):
        doc = "Test document content"
        result = split_document_into_sections(doc, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], doc)

    def test_invalid_target_too_low(self):
        with self.assertRaises(ValueError):
            split_document_into_sections("test", 0)

    def test_invalid_target_too_high(self):
        with self.assertRaises(ValueError):
            split_document_into_sections("test", 51)

    def test_empty_document(self):
        with self.assertRaises(ValueError):
            split_document_into_sections("", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
