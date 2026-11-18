from worker.nas_search import MAX_IN_PROCESS_SEARCH_GROWTH_MB, MAX_SEARCH_RSS_MB, search_rss_cap_mb


def test_subprocess_search_uses_the_absolute_cap() -> None:
    assert search_rss_cap_mb(3900, in_process=False) == MAX_SEARCH_RSS_MB


def test_in_process_search_caps_growth_above_the_service_footprint() -> None:
    # A long-lived service already above the absolute cap may still search.
    assert search_rss_cap_mb(3900, in_process=True) == 3900 + MAX_IN_PROCESS_SEARCH_GROWTH_MB
    # A fresh service keeps at least the absolute cap.
    assert search_rss_cap_mb(500, in_process=True) == MAX_SEARCH_RSS_MB
