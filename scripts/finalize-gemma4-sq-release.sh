#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
model_dir="${repo_root}/models/gemma4-96e-a4b-heretic"
upload_dir="${repo_root}/models/hf-upload-gemma4-sq4"

sq4_model="${model_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf"
sq3_model="${model_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
spectral_kv="${model_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf"
spectral_kv_gpu="${model_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gpu.gguf"
readme_path="${upload_dir}/README.md"

kv_source="${KV_CALIB_SOURCE:-}"
run_bench="${RUN_BENCH:-0}"
upload_release="${UPLOAD_RELEASE:-0}"
bench_output="${BENCH_OUTPUT:-${model_dir}/gemma4-sq-bench.json}"

if [[ ! -f "${sq4_model}" || ! -f "${sq3_model}" ]]; then
	printf 'missing SQ release models in %s\n' "${model_dir}" >&2
	exit 1
fi

if [[ -z "${kv_source}" ]]; then
	if [[ -f "${spectral_kv}" ]]; then
		kv_source="${spectral_kv}"
	elif [[ -f "${spectral_kv_gpu}" ]]; then
		kv_source="${spectral_kv_gpu}"
	else
		printf 'missing SKV3 calibration sidecar, checked %s and %s\n' "${spectral_kv}" "${spectral_kv_gpu}" >&2
		exit 1
	fi
fi

if [[ ! -f "${kv_source}" ]]; then
	printf 'missing KV calibration source: %s\n' "${kv_source}" >&2
	exit 1
fi

render_upload_readme() {
	python3 - "${readme_path}" "${sq4_model}" "${sq3_model}" "${spectral_kv}" <<'PY'
import re
import sys
from pathlib import Path

readme_path = Path(sys.argv[1])
artifacts = {
    "SQ4_1S": ("Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf", Path(sys.argv[2])),
    "SQ3_1S": ("Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf", Path(sys.argv[3])),
    "SKV3": ("Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf", Path(sys.argv[4])),
}

text = readme_path.read_text(encoding="utf-8")

for label, (filename, path) in artifacts.items():
    size_bytes = path.stat().st_size
    size_line = f"Size: `{size_bytes:,}` bytes (`~{size_bytes / (1024 ** 3):.2f} GiB`)"
    placeholder = rf"Size: `SIZE_{label}_BYTES` bytes \(`SIZE_{label}_GIB`\)"
    concrete = rf"Size: `[^`]+` bytes \(`[^`]+`\)"
    section = re.compile(rf"(`{re.escape(filename)}`\n- )" + concrete)
    if re.search(placeholder, text):
        text = re.sub(placeholder, size_line, text)
    elif section.search(text):
        text = section.sub(rf"\1{size_line}", text)

readme_path.write_text(text, encoding="utf-8")
PY
}

if [[ "${kv_source}" != "${spectral_kv}" ]]; then
	ln -f "${kv_source}" "${spectral_kv}"
fi
"${repo_root}/scripts/sync-gemma4-chat-templates.sh"
REPACK_ONLY=sq "${repo_root}/scripts/repack-gemma4-chat-metadata.sh"
if [[ ! -e "${upload_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf" ]] || \
	[[ "$(stat -f '%i' "${upload_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf" 2>/dev/null || echo missing)" != "$(stat -f '%i' "${spectral_kv}")" ]]; then
	ln -f "${spectral_kv}" "${upload_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf"
fi
render_upload_readme

if [[ "${run_bench}" == "1" ]]; then
	python3 "${repo_root}/scripts/benchmark-gemma4-sq-matrix.py" \
		--spectral-calibration "${spectral_kv}" \
		--output-json "${bench_output}"
fi

if [[ "${upload_release}" == "1" ]]; then
	"${repo_root}/scripts/upload-gemma4-sq-to-hf.sh"
fi

printf 'finalized SQ release assets:\n'
printf '  %s\n' "${sq4_model}"
printf '  %s\n' "${sq3_model}"
printf '  %s\n' "${spectral_kv}"
printf '  %s\n' "${upload_dir}"
