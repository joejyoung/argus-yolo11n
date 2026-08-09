#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# argus_build_engine.sh — ONNX -> TensorRT FP16 engine, on-device.
#
# Stage 2 of the deployment path:
#
#     yolo11n.pt -> ONNX -> [ trtexec ] -> FP16 engine (sm_87) -> inference
#
# PURPOSE
#   Wraps trtexec and, just as importantly, records the machine state the
#   engine was built under. A TensorRT engine is not portable: it is compiled
#   for one compute capability, one TensorRT version, one driver. An engine
#   built anywhere but the Orin Nano will not load on the Orin Nano, so this
#   script must run on the device.
#
#   The build log is a presentation artifact, not a throwaway. Every run writes
#   a timestamped log containing the power mode, a tegrastats sample, the
#   TensorRT and JetPack versions, the ONNX checksum, and the complete verbose
#   trtexec output. A latency figure produced later is only meaningful next to
#   the power state recorded here.
#
#   Needs no Python environment. trtexec is a compiled binary; the project venv
#   is irrelevant to this step and activating it will not fix a failure here.
#
# USAGE
#   ./argus_build_engine.sh
#   ./argus_build_engine.sh runs/yolo11n.onnx runs/yolo11n_fp16.engine
#   ./argus_build_engine.sh --workspace 4096 --force
#   ./argus_build_engine.sh --dry-run            # print the command, build nothing
#   ./argus_build_engine.sh -- --best            # pass extra flags to trtexec
#
# ENVIRONMENT
#   ARGUS_ONNX        default ONNX input      (runs/yolo11n.onnx)
#   ARGUS_ENGINE      default engine output   (runs/yolo11n_fp16.engine)
#   ARGUS_WORKSPACE   workspace pool in MB    (2048)
#   ARGUS_TRTEXEC     trtexec path            (/usr/src/tensorrt/bin/trtexec)
#   ARGUS_RUNDIR      log destination         (runs)
#
# EXIT CODES
#   0  engine built and present on disk
#   1  trtexec failed, or produced no engine
#   2  prerequisite missing (trtexec, ONNX) or bad arguments
#
# Records no hostname anywhere: 'uname -srm' rather than 'uname -a', by
# deliberate choice. This repo goes public and a later scrub is more expensive
# than getting it right now.
# ---------------------------------------------------------------------------

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ONNX="${ARGUS_ONNX:-runs/yolo11n.onnx}"
ENGINE="${ARGUS_ENGINE:-runs/yolo11n_fp16.engine}"
WORKSPACE_MB="${ARGUS_WORKSPACE:-2048}"
TRTEXEC="${ARGUS_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
RUNDIR="${ARGUS_RUNDIR:-runs}"

FORCE=0
DRY_RUN=0
EXTRA=()

C_OK=$'\033[0;32m'; C_WARN=$'\033[0;33m'; C_ERR=$'\033[0;31m'
C_HDR=$'\033[1;36m'; C_OFF=$'\033[0m'
[[ -t 1 ]] || { C_OK=""; C_WARN=""; C_ERR=""; C_HDR=""; C_OFF=""; }

ok()   { printf '  %s[ OK ]%s   %s\n' "$C_OK"   "$C_OFF" "$*"; }
warn() { printf '  %s[WARN]%s   %s\n' "$C_WARN" "$C_OFF" "$*"; }
err()  { printf '  %s[FAIL]%s   %s\n' "$C_ERR" "$C_OFF" "$*" >&2; }
hdr()  { printf '\n%s=== %s %s\n' "$C_HDR" "$*" "$C_OFF"; }

usage() { sed -n '3,46p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)  WORKSPACE_MB="$2"; shift 2 ;;
        --trtexec)    TRTEXEC="$2";      shift 2 ;;
        --rundir)     RUNDIR="$2";       shift 2 ;;
        --force)      FORCE=1;           shift ;;
        --dry-run)    DRY_RUN=1;         shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -*)           err "unknown option: $1"; usage >&2; exit 2 ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 1 ]] && ONNX="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && ENGINE="${POSITIONAL[1]}"
[[ ${#POSITIONAL[@]} -gt 2 ]] && { err "expected at most 2 paths"; exit 2; }

if [[ ! "$WORKSPACE_MB" =~ ^[0-9]+$ ]]; then
    err "--workspace wants an integer in MB, got '$WORKSPACE_MB'"; exit 2
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
hdr "Preflight"

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

[[ -f "$ONNX" ]] || { err "ONNX not found: $ONNX"; exit 2; }
ok "onnx          $ONNX ($(du -h "$ONNX" | cut -f1))"

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
log "precision           fp16"
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
CMD=(
    "$TRTEXEC"
    "--onnx=$ONNX"
    "--saveEngine=$ENGINE"
    --fp16
    "--memPoolSize=workspace:$WORKSPACE_MB"
    --verbose
)
[[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")

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
# trtexec can report success and still leave nothing behind if the save path
# was unwritable, so the presence of the file is checked independently of the
# exit status.
# ---------------------------------------------------------------------------
hdr "Result"

TRT_VER="$(grep -m1 -oE 'TensorRT version: [0-9.]+' "$LOG" | cut -d' ' -f3)"
[[ -n "$TRT_VER" ]] && log "tensorrt_reported   $TRT_VER"
log "build_seconds       $ELAPSED"
log "trtexec_status      $STATUS"

if [[ "$STATUS" -ne 0 ]]; then
    err "trtexec exited $STATUS after ${ELAPSED}s"
    printf '           %s\n' \
        "read $LOG before changing anything." \
        "Do not switch power mode or run jetson_clocks to make this pass." >&2
    exit 1
fi

if [[ ! -s "$ENGINE" ]]; then
    err "trtexec reported success but $ENGINE is missing or empty"
    exit 1
fi

log "engine_bytes        $(stat -c %s "$ENGINE")"
log "engine_sha256       $(sha256sum "$ENGINE" | cut -d' ' -f1)"

ok "engine        $ENGINE ($(du -h "$ENGINE" | cut -f1)) in ${ELAPSED}s"
ok "log           $LOG"
printf '\n           %s\n' \
    "Next: run argus_trt_infer.py --mode bench from the project venv." \
    "The engine is sm_87-specific and will not load on the workstation."
