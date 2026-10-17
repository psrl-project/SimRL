"""Tests for the central `srl._logging` setup."""

from __future__ import annotations

import logging

from srl._logging import setup_logging, srl_logger


class TestSetupLogging:
    def test_returns_the_shared_srl_logger(self):
        logger = setup_logging(force=True)
        assert logger is srl_logger
        assert logger.name == "srl"

    def test_idempotent_by_default(self):
        setup_logging(force=True)
        n_handlers = len(srl_logger.handlers)
        setup_logging()  # no force -> should not add more handlers.
        assert len(srl_logger.handlers) == n_handlers

    def test_force_reconfigures(self):
        setup_logging(level="DEBUG", force=True)
        assert srl_logger.level == logging.DEBUG
        setup_logging(level="WARNING", force=True)
        assert srl_logger.level == logging.WARNING

    def test_formatter_includes_timestamp_and_source_location(self):
        setup_logging(force=True)
        handler = srl_logger.handlers[0]
        fmt = handler.formatter._fmt
        assert "%(asctime)s" in fmt
        assert "%(filename)s" in fmt
        assert "%(lineno)d" in fmt
        assert "%(levelname)s" in fmt

    def test_does_not_propagate_to_root(self):
        setup_logging(force=True)
        assert srl_logger.propagate is False

    def test_string_level_is_case_insensitive(self):
        setup_logging(level="debug", force=True)
        assert srl_logger.level == logging.DEBUG

    def test_log_file_also_writes_to_disk(self, tmp_path):
        log_file = tmp_path / "run.log"
        setup_logging(level="INFO", log_file=log_file, force=True)
        srl_logger.info("hello from a test")
        for handler in srl_logger.handlers:
            handler.flush()
        assert log_file.exists()
        assert "hello from a test" in log_file.read_text()

    def test_env_var_default_level(self, monkeypatch):
        monkeypatch.setenv("SIMRL_LOG_LEVEL", "ERROR")
        setup_logging(force=True)
        assert srl_logger.level == logging.ERROR
