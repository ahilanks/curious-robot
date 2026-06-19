#!/usr/bin/env bash
# First-time setup for the Curious Robot training environment.
# Run once on a new pod: bash setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

SCENE_XML="${PROJECT_DIR}/env/SO101/scene.xml"

# .env holds WANDB_/HF_/GH_ creds; the training code also auto-loads it via python-dotenv.
load_env() {
    if [[ -f "${PROJECT_DIR}/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "${PROJECT_DIR}/.env"
        set +a
    fi
}
load_env

# Headless rendering: MuJoCo defaults to the on-screen GLFW backend, which needs an X11
# display. The MAIN process renders under OSMesa (CPU offscreen) -- it shares its process
# with the trainer's CUDA context, and an EGL render() in the CUDA process can SIGABRT once
# CUDA work accumulates. The subproc render workers are CUDA-free and default to GPU EGL
# (~100x faster; see --render-backend / env/parallel_env.py), which coexists with CUDA fine
# across processes. So this only sets the main proc's backend. Respect an explicit override.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

mkdir -p runs "${WANDB_DIR:-runs/wandb}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }
have_egl_loader() { ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; }
have_osmesa_loader() { ldconfig -p 2>/dev/null | grep -q 'libOSMesa\.so'; }

require_env() {
    local missing=0
    for key in WANDB_API_KEY HF_TOKEN HF_UPLOAD_REPO_ID GH_TOKEN; do
        if [[ -z "${!key:-}" ]]; then
            echo "  ERROR: required env var $key is not set (see .env)"
            missing=1
        fi
    done
    if [[ $missing -ne 0 ]]; then
        echo "  Fill in .env before running setup."
        exit 1
    fi
}

ensure_node_tooling() {
    export NVM_DIR="${HOME}/.nvm"
    if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
        echo "Installing nvm"
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
    fi
    set +u
    # shellcheck disable=SC1091
    . "${NVM_DIR}/nvm.sh"
    [[ -s "${NVM_DIR}/bash_completion" ]] && . "${NVM_DIR}/bash_completion"
    nvm install --lts
    nvm use --lts
    nvm alias default 'lts/*'
    set -u
    echo "Installing Codex CLI"
    npm install -g @openai/codex
}

ensure_mujoco_runtime_deps() {
    if have_osmesa_loader; then
        echo "MuJoCo runtime libraries already installed"
        return
    fi
    echo "Installing MuJoCo runtime libraries (OSMesa CPU rendering + GL)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y libosmesa6 libgl1 libegl1 libgles2
}

ensure_gh_cli() {
    if have_cmd gh; then return; fi
    echo "Installing GitHub CLI"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y gh
}

ensure_tmux() {
    if have_cmd tmux; then return; fi
    echo "Installing tmux"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y tmux
}

ensure_bwrap() {
    if have_cmd bwrap; then return; fi
    echo "Installing bwrap"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y bubblewrap
}

ensure_hf_cli() {
    if have_cmd hf; then return; fi
    echo "Installing Hugging Face CLI"
    python3 -m pip install -U "huggingface_hub[cli]"
}

ensure_claude_cli() {
    export PATH="${HOME}/.local/bin:${PATH}"
    if have_cmd claude; then return; fi
    echo "Installing Claude CLI"
    curl -fsSL https://claude.ai/install.sh | bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
}

ensure_mujoco_python() {
    if python3 -c "import mujoco" >/dev/null 2>&1; then return; fi
    echo "Installing MuJoCo Python package"
    python3 -m pip install mujoco
}

configure_optional_auth() {
    if have_cmd gh; then
        if [[ -n "${GH_TOKEN:-}" ]]; then
            printf '%s' "$GH_TOKEN" | gh auth login --with-token >/dev/null 2>&1 || true
        fi
        gh auth setup-git || true
        gh auth status || true
    fi
    if have_cmd hf; then
        if [[ -n "${HF_TOKEN:-}" ]]; then
            hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
        fi
        hf auth whoami || true
    fi
    if have_cmd wandb; then
        if [[ -n "${WANDB_API_KEY:-}" ]]; then
            wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
        fi
    fi
}

ensure_scene_xml() {
    if [[ ! -f "${SCENE_XML}" ]]; then
        echo "  ERROR: scene XML missing at ${SCENE_XML}"
        return 1
    fi
    echo "  scene XML present at ${SCENE_XML}"
}

echo "=== Curious Robot Setup ==="

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"
if python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "  OK (>= 3.10)"
else
    echo "  ERROR: Python >= 3.10 required"
    exit 1
fi

require_env

echo ""
echo "=== Installing dependencies ==="
ensure_mujoco_runtime_deps
python3 -m pip install --upgrade pip setuptools wheel
if ! python3 -m pip install -e ".[dev]"; then
    echo "ERROR: pip install -e \".[dev]\" failed"
    exit 1
fi

echo ""
echo "=== Installing CLI tooling ==="
ensure_node_tooling
ensure_tmux
ensure_gh_cli
ensure_bwrap
ensure_hf_cli
ensure_claude_cli

echo ""
echo "=== Auth Check ==="
configure_optional_auth

echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU memory: {props.total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected')
"

echo ""
echo "=== MuJoCo Check ==="
ensure_mujoco_python
SCENE_XML_PATH="${SCENE_XML}" python3 -c "
import os
import mujoco
xml = os.environ['SCENE_XML_PATH']
print(f'  MuJoCo version: {mujoco.__version__}')
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=64, width=64)
r.update_scene(d, camera=0)
frame = r.render()
r.close()
print(f'  scene loaded: {m.nbody} bodies, {m.ngeom} geoms, {m.ncam} cameras')
print(f'  renderer OK [{os.environ.get(\"MUJOCO_GL\")}]: {frame.shape}')
"

echo ""
echo "=== Scene XML Check ==="
ensure_scene_xml

echo ""
echo "=== Smoke Test ==="
WANDB_MODE=disabled python3 src/train.py --name smoke --n-envs 4 --total-steps 60 \
    --start-steps 5 --batch-size 16 --wm-batch-size 16 \
    --no-wandb --no-hf --save-every 0 --video-every 0

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Notes:"
echo "  - .env is auto-loaded by setup and the training code (python-dotenv)"
echo "  - Run training:   python src/train.py --name <run> --n-envs 8 --env-threads 8"
echo "  - Play a policy:  python src/play_policy.py --name <run>   (pulls latest ckpt from HF)"
echo "  - Eval dynamics:  python src/eval_predictor.py --ckpt runs/<run>/ckpt_*.pt"
