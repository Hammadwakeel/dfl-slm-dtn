from app.services.initial_models import _normalization_matches


def test_normalization_tolerates_recomputed_statistics_drift() -> None:
    stored = {"method": "standard", "mean": [0.13065278695572455], "standard_deviation": [0.30810049504139475]}
    current = {"method": "standard", "mean": [0.13065871856938918], "standard_deviation": [0.3081148988097204]}
    assert _normalization_matches(stored, current)


def test_normalization_rejects_real_changes() -> None:
    stored = {"method": "standard", "mean": [0.1307], "standard_deviation": [0.3081]}
    assert not _normalization_matches(stored, {"method": "minmax"})
    assert not _normalization_matches(stored, {**stored, "mean": [0.5]})
    assert not _normalization_matches(stored, {**stored, "mean": [0.1307, 0.2]})
