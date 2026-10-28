#!/usr/bin/env bash
# Forwards to the full validate -> publish pipeline (STUDY_3D_HRES.md §10).
exec bash "$(cd "$(dirname "$0")" && pwd)/start_smoke_r2_validate_pipeline.sh" "$@"
