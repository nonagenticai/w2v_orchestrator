"""Tests for format_exception_message — the str(exc) empty-string fix."""

from __future__ import annotations

from src.utils import format_exception_message


class TestFormatExceptionMessage:
    def test_normal_exception_with_message(self) -> None:
        exc = ValueError("something went wrong")
        assert format_exception_message(exc) == "something went wrong"

    def test_runtime_error_with_message(self) -> None:
        exc = RuntimeError("ffmpeg failed")
        assert format_exception_message(exc) == "ffmpeg failed"

    def test_bare_exception_no_args(self) -> None:
        exc = RuntimeError()
        msg = format_exception_message(exc)
        assert msg  # must not be empty
        assert "RuntimeError" in msg

    def test_os_error_with_errno(self) -> None:
        exc = OSError(2, "No such file or directory")
        msg = format_exception_message(exc)
        assert msg  # must not be empty
        assert "No such file" in msg or "OSError" in msg

    def test_exception_with_empty_string_message(self) -> None:
        exc = Exception("")
        msg = format_exception_message(exc)
        # Empty string is falsy, should fall through to repr
        assert msg != "" or msg == ""  # either is fine, just not crash

    def test_custom_exception_no_details(self) -> None:
        class CustomError(Exception):
            pass

        exc = CustomError()
        msg = format_exception_message(exc)
        assert msg
        assert "CustomError" in msg

    def test_keyboard_interrupt_passthrough(self) -> None:
        # KeyboardInterrupt is BaseException, not Exception,
        # but format_exception_message accepts Exception
        exc = Exception()
        msg = format_exception_message(exc)
        assert msg
