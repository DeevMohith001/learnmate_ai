from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

from data_ingestion.data_logger import ensure_log_files, load_json_records, log_chat_event, log_quiz_attempt, log_user_activity
from learnmate_ai.config import AppConfig
from learnmate_ai.database_manager import database_status
from learnmate_ai.storage import ensure_data_directories
from modules import analytics, chatbot_rag, vectorstore


class UploadStub:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class LearnMateSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path("tests/.tmp_runtime")
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.config = AppConfig(
            base_dir=self.temp_root,
            data_dir=self.temp_root / "data",
            raw_dir=self.temp_root / "data" / "raw",
            bronze_dir=self.temp_root / "data" / "bronze",
            silver_dir=self.temp_root / "data" / "silver",
            gold_dir=self.temp_root / "data" / "gold",
            report_dir=self.temp_root / "data" / "reports",
            logs_dir=self.temp_root / "data" / "logs",
            streaming_input_dir=self.temp_root / "data" / "stream_input",
            streaming_output_dir=self.temp_root / "data" / "stream_output",
            checkpoint_dir=self.temp_root / "data" / "checkpoints",
            mysql_password="",
        )
        ensure_data_directories(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_load_structured_data_valid_csv(self):
        upload = UploadStub("sample.csv", b"category,value\nA,1\nB,2\n")
        df = analytics.load_structured_data(upload)
        self.assertEqual(len(df), 2)

    def test_blank_chat_question(self):
        self.assertIn("Please enter", chatbot_rag.chatbot_respond("   "))

    def test_vectorstore_writes_json_text_file(self):
        temp_dir = self.temp_root / "vectorstore"
        temp_dir.mkdir(parents=True, exist_ok=True)
        index_path = str(temp_dir / "vectordb")
        vectorstore.build_vectorstore(["alpha beta", "gamma delta"], index_path=index_path)
        text_path = Path(f"{index_path}_texts.json")
        self.assertTrue(text_path.exists())
        with text_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        self.assertEqual(payload, ["alpha beta", "gamma delta"])

    def test_loggers_write_json_lines(self):
        ensure_log_files(self.config)
        log_quiz_attempt("7", "python", 86.5, 5, "quiz-1", self.config)
        log_chat_event("7", "python", "What is Spark?", "Spark is a distributed engine.", self.config)
        log_user_activity("7", "python", "summary_requested", {"mode": "brief"}, config=self.config)

        quiz_records = load_json_records(self.config.logs_dir / "quiz_logs.json")
        chat_records = load_json_records(self.config.logs_dir / "chat_logs.json")
        activity_records = load_json_records(self.config.logs_dir / "user_activity.json")

        self.assertEqual(quiz_records[0]["topic"], "python")
        self.assertEqual(chat_records[0]["action_type"], "chat_message")
        self.assertEqual(activity_records[0]["metadata"]["mode"], "brief")

    def test_database_status_reports_unconfigured_mysql(self):
        status = database_status(self.config)
        self.assertFalse(status["connected"])
        self.assertFalse(status["database_configured"])


if __name__ == "__main__":
    unittest.main()
