#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# argus_build_engine.sh — ONNX -> TensorRT engine, on-device.
#
# Stage 2 of the deployment path:
#
#     yolo11n.pt -> ONNX -> [ this script ] -> FP16 or INT8 engine (sm_87)
#
# PURPOSE
#   Builds one engine for a static imgsz of 640 or 1280, in FP16 or INT8.
#   FP16 is trtexec --fp16. INT8 calls argus_build_int8.py, which calibrates
#   from images in --calib-dir and falls back to FP16 for layers that have no
#   INT8 kernel. An engine built anywhere but the Orin Nano will not load on
#   the Orin Nano, so this script must run on the device.
#
#   The build log is a presentation artifact. Every run writes a timestamped
#   log containing the power mode, a tegrastats sample, the TensorRT and
#   JetPack versions, the ONNX checksum, and the builder output. A latency
#   figure produced later is only meaningful next to the power state recorded
#   here.
#
#   FP16 needs no Python environment. INT8 needs the project venv active so
#   import tensorrt resolves. Do not pip install tensorrt.
#
# USAGE
#   ./argus_build_engine.sh
#   ./argus_build_engine.sh --precision fp16 --imgsz 640
#   ./argus_build_engine.sh --precision int8 --imgsz 1280 --calib-dir runs/calib
#   ./argus_build_engine.sh runs/yolo11n_640.onnx runs/yolo11n_fp16_640.engine
#   ./argus_build_engine.sh --workspace 4096 --force
#   ./argus_build_engine.sh --dry-run
#   ./argus_build_engine.sh -- --best          # extra flags, FP16 / trtexec only
#
# ENVIRONMENT
#   ARGUS_ONNX        ONNX input (default runs/yolo11n_<imgsz>.onnx)
#   ARGUS_ENGINE      engine output (default runs/yolo11n_<precision>_<imgsz>.engine)
#   ARGUS_PRECISION   fp16 or int8 (default fp16)
#   ARGUS_IMGSZ       640 or 1280 (default 640)
#   ARGUS_CALIB_DIR   INT8 calibration images (default runs/calib)
#   ARGUS_WORKSPACE   workspace pool in MB (2048)
#   ARGUS_TRTEXEC     trtexec path (/usr/src/tensorrt/bin/trtexec)
#   ARGUS_PYTHON      python for the INT8 builder (python3)
#   ARGUS_RUNDIR      log destination (runs)
#
# EXIT CODES
#   0  engine built and present on disk
#   1  the builder failed, or produced no engine
#   2  prerequisite missing or bad arguments
#
# Records no hostname anywhere: 'uname -srm' rather than 'uname -a', by
# deliberate choice. This repo goes public and a later scrub is more expensive
# than getting it right now.
# ---------------------------------------------------------------------------

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRECISION="${ARGUS_PRECISION:-fp16}"
IMGSZ="${ARGUS_IMGSZ:-640}"
CALIB_DIR="${ARGUS_CALIB_DIR:-runs/calib}"
WORKSPACE_MB="${ARGUS_WORKSPACE:-2048}"
TRTEXEC="${ARGUS_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PYTHON="${ARGUS_PYTHON:-python3}"
RUNDIR="${ARGUS_RUNDIR:-runs}"

FORCE=0
DRY_RUN=0
EXTRA=()
ONNX_ARG=""
ENGINE_ARG=""

C_OK=$'\033[0;32m'; C_WARN=$'\033[0;33m'; C_ERR=$'\033[0;31m'
C_HDR=$'\033[1;36m'; C_OFF=$'\033[0m'
[[ -t 1 ]] || { C_OK=""; C_WARN=""; C_ERR=""; C_HDR=""; C_OFF=""; }

ok()   { printf '  %s[ OK ]%s   %s\n' "$C_OK"   "$C_OFF" "$*"; }
warn() { printf '  %s[WARN]%s   %s\n' "$C_WARN" "$C_OFF" "$*"; }
err()  { printf '  %s[FAIL]%s   %s\n' "$C_ERR" "$C_OFF" "$*" >&2; }
hdr()  { printf '\n%s=== %s %s\n' "$C_HDR" "$*" "$C_OFF"; }

usage() { sed -n '3,52p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision)  PRECISION="$2";    shift 2 ;;
        --imgsz)      IMGSZ="$2";        shift 2 ;;
        --calib-dir)  CALIB_DIR="$2";    shift 2 ;;
        --workspace)  WORKSPACE_MB="$2"; shift 2 ;;
        --trtexec)    TRTEXEC="$2";      shift 2 ;;
        --python)     PYTHON="$2";       shift 2 ;;
        --rundir)     RUNDIR="$2";       shift 2 ;;
        --force)      FORCE=1;           shift ;;
        --dry-run)    DRY_RUN=1;         shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -*)           err "unknown option: $1"; usage >&2; exit 2 ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 1 ]] && ONNX_ARG="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && ENGINE_ARG="${POSITIONAL[1]}"
[[ ${#POSITIONAL[@]} -gt 2 ]] && { err "expected at most 2 paths"; exit 2; }

case "$PRECISION" in
    fp16|int8) ;;
    *) err "--precision wants fp16 or int8, got '$PRECISION'"; exit 2 ;;
esac
case "$IMGSZ" in
    640|1280) ;;
    *) err "--imgsz wants 640 or 1280, got '$IMGSZ'"; exit 2 ;;
esac
if [[ ! "$WORKSPACE_MB" =~ ^[0-9]+$ ]]; then
    err "--workspace wants an integer in MB, got '$WORKSPACE_MB'"; exit 2
fi
if [[ "$PRECISION" == "int8" && ${#EXTRA[@]} -gt 0 ]]; then
    err "extra trtexec flags are only valid with --precision fp16"
    exit 2
fi

if [[ -n "$ONNX_ARG" ]]; then
    ONNX="$ONNX_ARG"
elif [[ -n "${ARGUS_ONNX:-}" ]]; then
    ONNX="$ARGUS_ONNX"
else
    ONNX="runs/yolo11n_${IMGSZ}.onnx"
fi

if [[ -n "$ENGINE_ARG" ]]; then
    ENGINE="$ENGINE_ARG"
elif [[ -n "${ARGUS_ENGINE:-}" ]]; then
    ENGINE="$ARGUS_ENGINE"
else
    ENGINE="runs/yolo11n_${PRECISION}_${IMGSZ}.engine"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INT8_PY="$SCRIPT_DIR/argus_build_int8.py"
CACHE="$RUNDIR/calib_int8_${IMGSZ}.cache"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
hdr "Preflight"

if [[ "$PRECISION" == "fp16" ]]; then
    if [[ ! -x "$TRTEXEC" ]]; then
        if command -v trtexec >/dev/null 2>&1; then
            TRTEXEC="$(command -v trtexec)"
        else
            err "trtexec not found at $TRTEXEC"
            printf '           %s\n' \
                "trtexec ships with the system TensorRT, normally at" \
                "/usr/src/tensorrt/bin/trtexec. Set ARGUS_TRTEXEC if it lives" \
                "elsewhere. Do not pip-install tensorrt to work around this." >&2
            exit 2
        fi
    fi
    ok "trtexec       $TRTEXEC"
else
    [[ -f "$INT8_PY" ]] || { err "INT8 builder not found: $INT8_PY"; exit 2; }
    command -v "$PYTHON" >/dev/null 2>&1 || { err "python not found: $PYTHON"; exit 2; }
    ok "int8 builder  $INT8_PY"
    ok "python        $PYTHON"
    ok "calib dir     $CALIB_DIR"
fi

[[ -f "$ONNX" ]] || { err "ONNX not found: $ONNX"; exit 2; }
ok "onnx          $ONNX ($(du -h "$ONNX" | cut -f1))"
ok "precision     $PRECISION"
ok "imgsz         $IMGSZ"

if [[ -e "$ENGINE" && "$FORCE" -eq 0 ]]; then
    err "engine already exists: $ENGINE"
    printf '           %s\n' "pass --force to rebuild over it" >&2
    exit 2
fi

mkdir -p "$RUNDIR" "$(dirname "$ENGINE")" || exit 2

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RUNDIR/engine_build_${STAMP}.log"

# Plain text to both the log and the console. Status lines above are coloured
# and console-only; nothing in the log carries escape sequences.
log() { printf '%s\n' "$*" | tee -a "$LOG" >/dev/null; printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# Provenance
#
# Captured before the build, because the point is to describe the machine that
# compiled this engine. Every probe is tolerant of absence: a missing tool
# should degrade the record, not abort the build.
# ---------------------------------------------------------------------------
probe() {
    local label="$1"; shift
    local out
    if out="$("$@" 2>/dev/null)" && [[ -n "$out" ]]; then
        printf '%s' "$out"
    else
        printf 'unavailable'
    fi
}

hdr "Provenance"

log "argus_build_engine.sh"
log "utc                 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "kernel              $(uname -srm)"
log "onnx                $ONNX"
log "onnx_sha256         $(sha256sum "$ONNX" | cut -d' ' -f1)"
log "onnx_bytes          $(stat -c %s "$ONNX")"
log "engine_out          $ENGINE"
log "precision           $PRECISION"
log "imgsz               $IMGSZ"
if [[ "$PRECISION" == "int8" ]]; then
    log "calib_dir           $CALIB_DIR"
    log "calib_cache         $CACHE"
fi
log "workspace_mb        $WORKSPACE_MB"

POWER="$(probe power nvpmodel -q)"
if [[ "$POWER" == "unavailable" ]]; then
    log "power_mode          unavailable"
    warn "nvpmodel -q failed. Every latency figure from this engine is void"
    warn "without a recorded power state -- record it by hand before benching."
else
    log "power_mode          $(printf '%s' "$POWER" | tr '\n' ' ' | sed 's/  */ /g')"
fi

log "tensorrt_apt        $(probe trt dpkg-query -W -f='${Version}' tensorrt)"
log "libnvinfer_apt      $(probe nvi dpkg-query -W -f='${Version}' libnvinfer10)"
log "jetpack_apt         $(probe jp dpkg-query -W -f='${Version}' nvidia-jetpack)"
if [[ -r /etc/nv_tegra_release ]]; then
    log "nv_tegra_release    $(head -n 1 /etc/nv_tegra_release)"
fi
log "cuda                $(probe cuda sh -c 'nvcc --version 2>/dev/null | tail -n 1')"

TEGRA="$(timeout 4 tegrastats --interval 1000 2>/dev/null | head -n 1)"
log "tegrastats          ${TEGRA:-unavailable}"
log ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [[ "$PRECISION" == "fp16" ]]; then
    BUILDER="trtexec"
    CMD=(
        "$TRTEXEC"
        "--onnx=$ONNX"
        "--saveEngine=$ENGINE"
        --fp16
        "--memPoolSize=workspace:$WORKSPACE_MB"
        --verbose
    )
    [[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")
else
    BUILDER="argus_build_int8.py"
    CMD=(
        "$PYTHON" "$INT8_PY"
        --onnx "$ONNX"
        --engine "$ENGINE"
        --imgsz "$IMGSZ"
        --workspace "$WORKSPACE_MB"
        --calib-dir "$CALIB_DIR"
        --cache "$CACHE"
    )
fi

log "command             ${CMD[*]}"
log ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    hdr "Dry run"
    ok "nothing built; provenance written to $LOG"
    exit 0
fi

hdr "Building"
printf '           %s\n' "this takes a few minutes; full output goes to $LOG"

SECONDS=0
"${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS="${PIPESTATUS[0]}"
ELAPSED="$SECONDS"

# ---------------------------------------------------------------------------
# Verify
#
# The builder can report success and still leave nothing behind if the save
# path was unwritable, so the presence of the file is checked independently
# of the exit status.
# ---------------------------------------------------------------------------
hdr "Result"

TRT_VER="$(grep -m1 -oE 'TensorRT version: [0-9.]+' "$LOG" | cut -d' ' -f3)"
[[ -n "$TRT_VER" ]] && log "tensorrt_reported   $TRT_VER"
log "build_seconds       $ELAPSED"
log "builder_status      $STATUS"

if [[ "$STATUS" -ne 0 ]]; then
    err "$BUILDER exited $STATUS after ${ELAPSED}s"
    printf '           %s\n' \
        "read $LOG before changing anything." \
        "Do not switch power mode or run jetson_clocks to make this pass." >&2
    exit 1
fi

if [[ ! -s "$ENGINE" ]]; then
    err "$BUILDER reported success but $ENGINE is missing or empty"
    exit 1
fi

log "engine_bytes        $(stat -c %s "$ENGINE")"
log "engine_sha256       $(sha256sum "$ENGINE" | cut -d' ' -f1)"

ok "engine        $ENGINE ($(du -h "$ENGINE" | cut -f1)) in ${ELAPSED}s"
ok "log           $LOG"
printf '\n           %s\n' \
    "Next: run argus_trt_infer.py --mode bench, or argus_live_demo.py," \
    "from the project venv. The engine will not load on the workstation."
