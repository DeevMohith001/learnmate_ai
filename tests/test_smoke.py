from __future__ import annotations

import unittest

from modules import analytics, chatbot_rag


class UploadStub:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class LearnMateSmokeTests(unittest.TestCase):
    def test_load_structured_data_valid_csv(self):
        upload = UploadStub("sample.csv", b"category,value\nA,1\nB,2\n")
        df = analytics.load_structured_data(upload)
        self.assertEqual(len(df), 2)

    def test_load_structured_data_empty_file(self):
        upload = UploadStub("sample.csv", b"")
        with self.assertRaises(ValueError):
            analytics.load_structured_data(upload)

    def test_blank_chat_question(self):
        self.assertIn("Please enter", chatbot_rag.chatbot_respond("   "))


if __name__ == "__main__":
    unittest.main()
