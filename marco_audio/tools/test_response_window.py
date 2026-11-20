"""No hardware required: python marco_audio/tools/test_response_window.py."""
import unittest
from response_window import ResponseWindow


class ReplyTests(unittest.TestCase):
    def test_playback_echo_guard_and_one_reply(self):
        window = ResponseWindow(seconds=10)
        token = window.request()
        self.assertIsNone(window.request())
        self.assertIsNone(window.capture_id(1))
        self.assertFalse(window.accept(token, 1, 2, 2))
        window.played(token, 3)
        self.assertIsNone(window.capture_id(3.2))
        self.assertEqual(window.capture_id(3.4), token)
        self.assertFalse(window.accept(token, 3.1, 3.7, 4))
        self.assertTrue(window.accept(token, 3.4, 3.8, 4.5))
        self.assertFalse(window.accept(token, 4, 4.5, 5))
        self.assertEqual(window.capture_id(5), -1)

    def test_timeout_and_late_decoder_result(self):
        window = ResponseWindow(seconds=2)
        token = window.request()
        window.played(token, 0)
        self.assertFalse(window.accept(token, 2, 2.4, 3))
        # The word ends in time, even though recognition completes later.
        self.assertTrue(window.accept(token, 1.7, 2.2, 3.5))
        token = window.request()
        window.played(token, 10)
        self.assertIn('No usable Polo', window.status(20))
        self.assertFalse(window.accept(token, 11, 12, 20))

    def test_failed_cancelled_and_previous_rounds(self):
        window = ResponseWindow()
        first = window.request()
        window.failed(first, 'Playback failed')
        self.assertEqual(window.capture_id(1), -1)
        window.played(first, 1)
        self.assertFalse(window.accept(first, 2, 3, 4))
        second = window.request()
        window.cancel()
        window.played(second, 5)
        third = window.request()
        window.played(third, 10)
        self.assertFalse(window.accept(second, 11, 12, 13))
        self.assertTrue(window.accept(third, 11, 12, 13))


if __name__ == '__main__':
    unittest.main()
