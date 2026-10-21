# Day 03: Data Engineering Pipeline — Walkthrough Report

**Project**: AI Operations Intelligence Platform  
**Branch**: `feature/day-03-data-pipeline`  
**Date**: October 2026  
**Status**: IN PROGRESS  

---

## 1. Timeline Metadata Contract (T1 Verification)

Inspection of `data/raw/unified/timeline_metadata.json` confirmed the following actual keys and values:

- **Global Timeline Parameters**:
  - `shift_days`: `2240` (forward shift applied to MetroPT timestamps)
  - `shift_weeks`: `320` (integer weeks preserving diurnal and weekly patterns)
  - `timeline_start`: `"2026-03-21"`
  - `timeline_end`: `"2026-10-19"`
  - `I_OFF`: `1.55` (motor current cutoff separating running from off/idle)

- **Machine Profiles (`machines` dictionary)**:
  - Keys: `M-101` through `M-112`.
  - Attributes per machine:
    - `transforms`: Resampling parameters (`current_scale`, `temp_scale`, `temp_offset`, `pressure_scale`).
    - `initial_operating_hours`: Baseline cumulative operating hours at timeline start.
    - `utilization_factor`: Target duty cycle fraction.
    - `failure_events`: Array of discrete failure dates:
      - `M-105`: `["2026-06-21"]` (Hydraulic seal blowout, Day 93)
      - `M-110`: `["2026-08-05"]` (Bearing overstrain seizure, Day 138)
      - `M-111`: `["2026-08-22"]` (Heat dissipation VFD burnout, Day 155)
      - Others: `[]` (empty list)
    - `window_overlay_applied`: `true` for `M-104` (denotes MetroPT real failure overlay and hero degradation).

This contract serves as the foundation for the evaluation label generator in T7.

---
