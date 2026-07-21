# =============================================================================
# batch env_file — ANNOTATED EXAMPLE (copy, edit, keep OUTSIDE the package)
# =============================================================================
#
# The batch job script sources this file on the compute node with a NON-login
# shell (`#!/bin/zsh`, no `-l`).  It is the single machine-specific artifact the
# `batch` package needs: the package carries only the *path* to your copy
# (`--env-file` or `$BATCH_ENV_FILE`), never the file itself.
#
# CONTRACT — after this file is sourced, the shell MUST have:
#   1. the pinned run modules loaded,
#   2. the venv python resolvable (and passed to batch as `--python`), and
#   3. SRC / CLAW / PYTHONPATH exported so that `python -c "import batch"`
#      exits 0.
# The job script runs exactly that import as a fail-fast check before the solver.
#
# RULES this file must respect:
#   * Be self-sufficient: source the Lmod init yourself if `module` is not a
#     function.  The batch path must NEVER depend on ~/.zprofile or ~/.bashrc.
#   * Only READ scheduler- / site-provided variables (PBS_*, SLURM_*, NCAR_*);
#     never redefine them.  batch itself exports the BATCH_* contract — leave
#     those alone here too.
#   * Guard an empty PYTHONPATH so you never emit a bare leading ":".
#
# This example targets NCAR Derecho; adapt the module names, paths, and venv for
# your machine.  Lines you must change are marked EDIT.
# -----------------------------------------------------------------------------

# --- 1. Make `module` available in a non-login shell -------------------------
# On a login shell Lmod is set up by /etc/profile.d; here we do it ourselves so
# the batch path never relies on a login profile having run.
if ! typeset -f module >/dev/null 2>&1; then
    if [[ -n "${LMOD_PKG:-}" && -f "${LMOD_PKG}/init/zsh" ]]; then
        source "${LMOD_PKG}/init/zsh"
    elif [[ -f /etc/profile.d/z00_lmod.sh ]]; then
        source /etc/profile.d/z00_lmod.sh
    fi
fi

# --- 2. Load the pinned run modules ------------------------------------------
# Pin exact versions so a submitted batch reproduces months later.  EDIT.
module --force purge 2>/dev/null || true
module load ncarenv/23.09 2>/dev/null || true
module load conda 2>/dev/null || true

# --- 3. Activate the venv / conda env holding batch + clawpack ---------------
# EDIT: point at the environment where `pip install -e .` installed batch.
export BATCH_VENV="${BATCH_VENV:-$HOME/venvs/geoclaw}"
if [[ -f "${BATCH_VENV}/bin/activate" ]]; then
    source "${BATCH_VENV}/bin/activate"
fi
# Pass this same interpreter to batch as `--python "${BATCH_VENV}/bin/python"`
# so the launch is immune to any module-load PATH reordering above.

# --- 4. Export the source / clawpack paths -----------------------------------
# EDIT these to your checkout locations.
export SRC="${SRC:-$HOME/src}"
export CLAW="${CLAW:-$SRC/clawpack}"

# --- 5. Put clawpack + batch on PYTHONPATH (guarding an empty PYTHONPATH) -----
# Note: PYTHONPATH — never PYTHON_PATH.  The ":+" guard avoids a leading colon
# (a bare ":" means "current directory", a subtle and dangerous default).
export PYTHONPATH="${CLAW}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${SRC}/batch${PYTHONPATH:+:${PYTHONPATH}}"

# --- 6. (optional) run-wide environment --------------------------------------
# OMP_NUM_THREADS is set per-job by batch (from --omp-num-threads); do not pin it
# here.  Put only genuinely machine-wide settings below.  EDIT as needed.
export MPICH_OFI_STARTUP_CONNECT=1

# -----------------------------------------------------------------------------
# Verify the contract locally (no scheduler needed) with a clean environment:
#
#   env -i HOME="$HOME" bash --noprofile --norc -c \
#     'source docs/env_file.example.zsh; python -c "import batch"'
#
# It should exit 0 once the EDIT lines point at a real venv + checkout.
# -----------------------------------------------------------------------------
