"""Test for scripts/approve_gameweek.py - mocks the HTTP call, no live
orchestrator needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from approve_gameweek import resume_gameweek


def test_resume_gameweek_posts_the_expected_body_and_returns_the_response():
    fake_response = MagicMock()
    fake_response.read.return_value = json.dumps({"approval_status": "approved"}).encode("utf-8")
    fake_response.__enter__.return_value = fake_response

    with patch("approve_gameweek.urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
        result = resume_gameweek(12, "approve", base_url="http://example:8000")

    assert result == {"approval_status": "approved"}
    sent_request = mock_urlopen.call_args[0][0]
    assert sent_request.full_url == "http://example:8000/resume"
    assert json.loads(sent_request.data) == {"gameweek": 12, "decision": "approve"}
