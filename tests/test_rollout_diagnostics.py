from noise_rl.rollout_diagnostics import WeightUpdateTracker


def test_weight_update_tracker_reports_active_and_completed_windows():
    tracker = WeightUpdateTracker()
    assert tracker.snapshot(now=10)["observed"] is False

    tracker.observe("Timer update_weights start", now=10)
    active = tracker.snapshot(now=12.5)
    assert active["observed"] is True
    assert active["active"] is True
    assert active["active_seconds"] == 2.5

    tracker.observe("Timer update_weights end (elapsed: 3.0s)", now=13)
    completed = tracker.snapshot(now=15)
    assert completed["active"] is False
    assert completed["last_duration_seconds"] == 3
    assert completed["last_ended_seconds_ago"] == 2
