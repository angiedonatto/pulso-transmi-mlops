create index if not exists data_quality_checks_run_id_idx
    on public.data_quality_checks (run_id);

create index if not exists metrics_by_station_model_id_idx
    on public.metrics_by_station (model_id);

create index if not exists metrics_by_station_station_id_idx
    on public.metrics_by_station (station_id);

create index if not exists model_versions_run_id_idx
    on public.model_versions (run_id);

create index if not exists predictions_model_id_idx
    on public.predictions (model_id);

create index if not exists predictions_run_id_idx
    on public.predictions (run_id);

create index if not exists predictions_station_id_idx
    on public.predictions (station_id);

create index if not exists station_time_features_observed_at_idx
    on public.station_time_features (observed_at);
