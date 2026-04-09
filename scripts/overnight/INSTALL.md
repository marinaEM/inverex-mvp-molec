# INVEREX Overnight Benchmark — Installation Guide

This installs a `launchd` job that fires the model-vs-signal benchmark on
**Wednesday 2026-04-08 at 14:00 local time** and survives reboots / power cycles.

## Files

```
scripts/overnight/
├── _shared.py                       # data loaders, evaluation, LODO loop
├── agent1_lightgbm_sweep.py         # 50-trial Optuna sweep over LightGBM
├── agent2_catboost.py               # 30-trial CatBoost sweep
├── agent3_alt_learners.py           # ElasticNet LR + Random Forest + XGBoost
├── agent4_pathway.py                # Gene vs ssGSEA vs singscore vs combined
├── agent5_stratified.py             # Per-drug-class / cancer-type / technology LODO
├── agent6_diagnostics.py            # Per-gene assoc, PCA, silhouette, pathway tests
├── agent7_report.py                 # Aggregator + final markdown report
├── run_overnight_benchmark.sh       # Orchestrator (caffeinate + parallel launch)
├── com.inverex.overnight.plist      # launchd job definition
└── INSTALL.md                       # this file
```

## How it works

1. **launchd** loads `com.inverex.overnight` at every login and at the calendar
   trigger (Wed 2026-04-08 14:00).
2. The orchestrator script has **two gates** that make it safe to invoke any time:
   - **Date gate**: exits immediately if current time < trigger time.
   - **Idempotency gate**: exits immediately if `completed.flag` exists.
3. When the gates open, it:
   - Acquires a `running.flag` (PID-based, prevents double-execution)
   - Wraps execution in `caffeinate -i -m -s` to prevent system sleep
   - Launches **agents 1-6 in parallel** (each in `pixi run`)
   - Waits for all 6 to complete (with per-agent success/fail flags)
   - Runs **Agent 7** (report generator) sequentially after agents 1-6 finish
   - Writes `completed.flag` when done

## What to expect

- **Wall-clock duration**: 1-3 hours depending on hyperparameter sweeps and CPU.
- **Outputs** in `results/overnight_model_signal_benchmark/`:
  - `summaries/all_runs.csv` — every individual run with all 8+ metrics
  - `summaries/best_by_method.csv`
  - `summaries/best_by_stratum.csv`
  - `reports/final_overnight_summary.md` — full markdown report
  - `reports/morning_takeaway.txt` — concise bullet summary
  - `plots/model_comparison.png`, `plots/pca_diagnostics.png`
  - `diagnostics/*.tsv` — signal heterogeneity metrics
  - `raw_metrics/*.tsv` — per-fold metrics for every run
  - `logs/*.log` — per-agent execution logs

---

## Installation (one-time)

```bash
# 1. Copy the plist into LaunchAgents (user-level, no sudo needed)
cp scripts/overnight/com.inverex.overnight.plist ~/Library/LaunchAgents/

# 2. Load it into launchd
launchctl load ~/Library/LaunchAgents/com.inverex.overnight.plist

# 3. Verify it's loaded
launchctl list | grep com.inverex.overnight
```

You should see something like:
```
-       0       com.inverex.overnight
```
(The first column shows PID — `-` means not currently running, which is correct.
The second column shows last exit code — `0` means OK.)

---

## Verification

### Test the orchestrator immediately (date gate will protect you)

```bash
./scripts/overnight/run_overnight_benchmark.sh
```

Expected output: `Current time is before trigger (Wed 2026-04-08 14:00). Exiting.`

Check the launcher log to confirm:
```bash
cat results/overnight_model_signal_benchmark/logs/launcher.log
```

### Force a dry run (override the date gate temporarily)

If you want to actually test the pipeline before Wednesday, edit the plist
trigger time and the `TRIGGER_EPOCH` line in `run_overnight_benchmark.sh` to
something in the next 5 minutes, reload the plist, and watch it run.

```bash
# Reload after editing
launchctl unload ~/Library/LaunchAgents/com.inverex.overnight.plist
launchctl load ~/Library/LaunchAgents/com.inverex.overnight.plist
```

**Don't forget to revert the trigger time afterwards.**

---

## Power cycles and travel

- **Machine off at trigger time, boots after**: launchd will fire the job
  shortly after boot. The script's date gate is now satisfied, so it runs.
- **Machine asleep at trigger time**: launchd wakes the job. If the system was
  idle-sleeping, it will run immediately. If you forced sleep with the lid
  closed and the machine is in deep sleep, it will run on next wake.
- **Machine on but in clamshell with no power**: launchd queues; runs when AC
  is restored.
- **Already ran**: `completed.flag` blocks re-runs. Safe to invoke many times.

---

## Manual control

```bash
# Check status
launchctl list | grep com.inverex.overnight

# View logs while it's running
tail -f results/overnight_model_signal_benchmark/logs/launcher.log
tail -f results/overnight_model_signal_benchmark/logs/agent1_stdout.log

# Stop the job permanently
launchctl unload ~/Library/LaunchAgents/com.inverex.overnight.plist
rm ~/Library/LaunchAgents/com.inverex.overnight.plist

# Reset and re-run (after a completed run)
rm -rf results/overnight_model_signal_benchmark/.flags
launchctl unload ~/Library/LaunchAgents/com.inverex.overnight.plist
launchctl load ~/Library/LaunchAgents/com.inverex.overnight.plist
./scripts/overnight/run_overnight_benchmark.sh
```

---

## Disabling the job after the run

Once the benchmark is finished and the report is in your hands:

```bash
launchctl unload ~/Library/LaunchAgents/com.inverex.overnight.plist
rm ~/Library/LaunchAgents/com.inverex.overnight.plist
```
