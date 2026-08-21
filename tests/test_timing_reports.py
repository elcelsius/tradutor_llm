import json
import logging
import time
from datetime import datetime, timezone

from tradutor.config import AppConfig
from tradutor.main import _write_timing_report


def test_timing_report_keeps_latest_and_timestamped_history(tmp_path) -> None:
    cfg = AppConfig(output_dir=tmp_path)
    started_at = datetime(2026, 8, 21, 12, 34, 56, 123456, tzinfo=timezone.utc)

    payload = _write_timing_report(
        cfg=cfg,
        source_slug="volume",
        command="traduz",
        input_kind="pdf",
        status="success",
        started_at=started_at,
        run_started=time.perf_counter() - 1,
        timings={"translate": 0.5},
        logger=logging.getLogger(__name__),
    )

    latest = tmp_path / "volume_timings.json"
    history = tmp_path / "timings" / "volume_20260821_123456_123456_success.json"

    assert payload is not None
    assert latest.exists()
    assert history.exists()
    assert json.loads(latest.read_text(encoding="utf-8")) == json.loads(
        history.read_text(encoding="utf-8")
    )
