#!/usr/bin/env bash
# =============================================================================
# cluster/run_analysis.sh — Run Full Analysis on ALICE HPC or Local
# =============================================================================

# Activate conda
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate rikken_ai 2>/dev/null || true

PYTHON="python"
for candidate in "$CONDA_PREFIX/bin/python" "$HOME/miniforge3/envs/rikken_ai/bin/python" "$HOME/miniconda3/envs/rikken_ai/bin/python"; do
    if [ -x "$candidate" ]; then
        PYTHON="$candidate"
        break
    fi
done

$PYTHON analysis/run_all_analysis.py | tee analysis_report.txt
