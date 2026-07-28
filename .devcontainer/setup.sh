#!/usr/bin/env bash
# Kernel-development environment. Verifies the toolchain rather than assuming it.
set -euo pipefail

echo "── toolchain ──────────────────────────────────────────────"
nvcc --version | tail -2 || { echo "no nvcc: this image has no CUDA toolkit"; exit 1; }
python -c "import torch; print(f'torch {torch.__version__} · cuda {torch.version.cuda}')"

# The check that catches the commonest failure: a PyTorch built against a
# different CUDA than the one on PATH. The op compiles, then refuses to load
# with an undefined symbol that names nothing useful.
python - <<'PY'
import subprocess, sys, torch
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
tag = f"release {torch.version.cuda}"
if tag not in nvcc:
    print(f"\n  WARNING: torch was built against CUDA {torch.version.cuda}, but nvcc")
    print( "  on PATH is a different release. Custom ops will compile and then fail")
    print( "  to load with an undefined symbol. Match them before debugging further.\n")
    sys.exit(0)
print(f"toolkit matches torch ({tag})")
PY

echo "── profiling ──────────────────────────────────────────────"
for t in ncu nsys; do
  command -v $t >/dev/null && echo "$t: $($t --version 2>&1 | head -1)" \
    || echo "$t: absent — install Nsight to profile rather than guess"
done

echo "── clickllm ───────────────────────────────────────────────"
pip install -q -e . 2>/dev/null || true
python -m clickllm.kernels && python -m clickllm.plan

cat <<'MSG'

Ready. The workflow this environment exists for:

  1. clickllm kernel scaffold my-kernel      # a package, not a fork
  2. …write the kernel…
  3. ncu --set full python bench.py          # measure, do not guess
  4. clickllm prove --against baseline.json  # did it change the OUTPUT?

Step 4 is the one that is usually skipped and the only one that can tell a
1.4× win from an unreviewed model change.
MSG
