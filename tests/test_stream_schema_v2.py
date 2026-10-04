from runtime_data_validation import RuntimeDataValidator, RuntimeValidationContext


def row(value="61.00", quality="observed", unit="passengers", version=2):
    return {"station_id": "03000", "observed_at": "2026-09-20T12:15:00Z",
            "released_at": "2026-10-03T23:19:15.124962Z", "schema_version": version,
            "measurement": {"unit": unit, "value": value, "quality": quality}}


def test_real_v2_payload_is_ingested_without_changing_raw_evidence():
    raw = row()
    result = RuntimeDataValidator().validate([raw], RuntimeValidationContext(known_station_ids={"03000"}))
    assert len(result.rows_to_write) == 1
    assert result.rows_to_write[0].demand == 61
    assert result.rows_to_write[0].station_id == "03000"
    assert "demand" not in raw
    assert all(event.raw_payload == raw for event in result.events if event.event_type == "row")
    assert not result.has_critical


def test_mixed_v1_v2_page_keeps_both_valid_observations():
    legacy = {"station_id": "02300", "observed_at": "2026-09-20T12:15:00Z", "demand": 546}
    result = RuntimeDataValidator().validate([legacy, row()])
    assert {(r.station_id, r.demand) for r in result.rows_to_write} == {("02300", 546), ("03000", 61)}


def test_missing_unknown_units_versions_and_noninteger_measurements_are_quarantined():
    for raw in [row(None, quality="missing"), row("1.2"), row("-1"), row("NaN"),
                row(True), row(unit="vehicles"), row(version=3), row(version=True),
                {**row(), "demand": 62}]:
        result = RuntimeDataValidator().validate([raw])
        assert not result.rows_to_write
        assert result.quarantined_rows >= 1
