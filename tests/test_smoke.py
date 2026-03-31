from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

from learnmate_ai.config import AppConfig
from learnmate_ai.sqlite_manager import (
    initialize_sqlite_schema,
    list_registered_users,
    register_user,
    reset_sqlite_engine,
    sqlite_status,
)
from modules import analytics, chatbot_rag, vectorstore


class UploadStub:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class LearnMateSmokeTests(unittest.TestCase):
    def tearDown(self):
        config = AppConfig(sqlite_db_path=Path("tests/.tmp_learnmate.db"))
        reset_sqlite_engine(config)
        sqlite_path = Path("tests/.tmp_learnmate.db")
        if sqlite_path.exists():
            sqlite_path.unlink(missing_ok=True)

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

    def test_vectorstore_writes_json_text_file(self):
        temp_dir = Path("tests/.tmp_vectorstore_write")
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            index_path = str(temp_dir / "vectordb")
            vectorstore.build_vectorstore(["alpha beta", "gamma delta"], index_path=index_path)
            text_path = Path(f"{index_path}_texts.json")
            self.assertTrue(text_path.exists())
            with text_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.assertEqual(payload, ["alpha beta", "gamma delta"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_vectorstore_retrieval_fallback(self):
        temp_dir = Path("tests/.tmp_vectorstore_search")
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            index_path = str(temp_dir / "vectordb")
            vectorstore.build_vectorstore(["python handles tables", "spark handles large datasets"], index_path=index_path)
            results = vectorstore.retrieve_relevant_chunks("tables", index_path=index_path, score_threshold=2.0)
            self.assertTrue(results)
            self.assertIn("python", results[0].lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_sqlite_signup_persists_user(self):
        config = AppConfig(sqlite_db_path=Path("tests/.tmp_learnmate.db"))
        initialize_sqlite_schema(config)
        result = register_user("Test User", "test@example.com", "secret12", config)
        users = list_registered_users(config)
        status = sqlite_status(config)
        self.assertTrue(result["stored"])
        self.assertTrue(status["connected"])
        self.assertEqual(users[0]["email"], "test@example.com")


if __name__ == "__main__":
    unittest.main()
