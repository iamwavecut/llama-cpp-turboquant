#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
model_dir="${repo_root}/models/gemma4-96e-a4b-heretic"

input_q8="${model_dir}/Gemma-4-96E-A4B-Heretic-Q8_0.gguf"
spectral_weights="${model_dir}/Gemma-4-96E-A4B-Heretic-spectral-weights.gguf"
spectral_kv="${model_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf"
spectral_kv_gpu="${model_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gpu.gguf"
output_sq3="${model_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
output_sq4="${model_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf"
recipe_sq3="${model_dir}/requant_recipe_sq3_1s.txt"
recipe_sq4="${model_dir}/requant_recipe_sq4_1s.txt"
bench_prompt_short="${model_dir}/bench_prompt_short.txt"
bench_prompt_full="${model_dir}/bench_prompt_full.txt"
upload_dir="${repo_root}/models/hf-upload-gemma4-sq4"
upload_readme="${upload_dir}/README.md"
spectral_max_rows="${SPECTRAL_MAX_ROWS:-1024}"
spectral_threads="${SPECTRAL_THREADS:-6}"
spectral_profile_set="${SPECTRAL_PROFILE_SET:-all}"
kv_calib_n_gpu_layers="${KV_CALIB_N_GPU_LAYERS:-0}"
kv_calib_device="${KV_CALIB_DEVICE:-}"
kv_calib_source="${KV_CALIB_SOURCE:-}"
force_rebuild="${FORCE_REBUILD:-0}"
quant_threads="${QUANT_THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 8)}"

if [[ ! -f "${input_q8}" ]]; then
	printf 'missing %s\n' "${input_q8}" >&2
	exit 1
fi

if [[ ! -f "${recipe_sq3}" || ! -f "${recipe_sq4}" ]]; then
	printf 'missing SQ requant recipes in %s\n' "${model_dir}" >&2
	exit 1
fi

build_match_regex() {
	python3 - "${recipe_sq3}" "${recipe_sq4}" <<'PY'
import re
import sys
from pathlib import Path

names = []
for recipe_path in sys.argv[1:]:
    for line in Path(recipe_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, qtype = line.split("=", 1)
        if qtype.strip().lower() in {"sq3_1s", "sq4_1s"}:
            names.append(re.escape(name.strip()))

deduped = sorted(set(names))
print("^(?:" + "|".join(deduped) + ")$")
PY
}

build_expanded_recipe() {
	local overlay_recipe="$1"
	local expanded_recipe="$2"

	python3 - "${repo_root}/build/bin/llama-gguf" "${input_q8}" "${overlay_recipe}" "${expanded_recipe}" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

gguf_bin = Path(sys.argv[1])
input_model = Path(sys.argv[2])
overlay_path = Path(sys.argv[3])
expanded_path = Path(sys.argv[4])

tensor_re = re.compile(r"tensor\[\d+\]: name = (.*?), size = .*?, offset = .*?, type = ([^, ]+),")
type_map = {}

proc = subprocess.run(
    [str(gguf_bin), str(input_model), "r", "n"],
    check=True,
    capture_output=True,
    text=True,
)
for line in proc.stdout.splitlines():
    m = tensor_re.search(line)
    if not m:
        continue
    name, tensor_type = m.group(1), m.group(2)
    type_map[name] = tensor_type.lower()

if not type_map:
    raise SystemExit("failed to extract tensor type map from input gguf")

for raw_line in overlay_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or "=" not in line:
        continue
    name, tensor_type = line.split("=", 1)
    name = name.strip()
    if name not in type_map:
        print(f"warning: skipping recipe entry for unknown tensor: {name}", file=sys.stderr)
        continue
    type_map[name] = tensor_type.strip().lower()

expanded_path.write_text(
    "".join(f"{name}={tensor_type}\n" for name, tensor_type in sorted(type_map.items())),
    encoding="utf-8",
)
PY
}

render_upload_readme() {
	python3 - "${upload_readme}" "${output_sq4}" "${output_sq3}" "${spectral_kv}" <<'PY'
import sys
from pathlib import Path

readme_path = Path(sys.argv[1])
artifacts = {
    "SIZE_SQ4_BYTES": Path(sys.argv[2]),
    "SIZE_SQ3_BYTES": Path(sys.argv[3]),
    "SIZE_SKV3_BYTES": Path(sys.argv[4]),
}

text = readme_path.read_text(encoding="utf-8")

def format_bytes(size: int) -> str:
    return f"{size:,}"

def format_gib(size: int) -> str:
    return f"~{size / (1024 ** 3):.2f} GiB"

for key, path in artifacts.items():
    size = path.stat().st_size
    text = text.replace(key, format_bytes(size))
    text = text.replace(key.replace("_BYTES", "_GIB"), format_gib(size))

readme_path.write_text(text, encoding="utf-8")
PY
}

run_sq_quantize() {
	local output_path="$1"
	local recipe_path="$2"
	local outtype="$3"
	local stamp_path="${output_path}.complete"

	if [[ -f "${output_path}" && "${force_rebuild}" != "1" ]]; then
		if [[ -f "${stamp_path}" ]]; then
			printf 'reusing existing %s\n' "${output_path}"
			return
		fi
		printf 'existing %s has no completion stamp, rebuilding\n' "${output_path}" >&2
	fi

	rm -f "${stamp_path}"
	rm -f "${output_path}"
	"${repo_root}/build/bin/llama-quantize" \
		--allow-requantize \
		--spectral-calibration "${spectral_weights}" \
		--spectral-profile all \
		--tensor-type-file "${recipe_path}" \
		"${input_q8}" \
		"${output_path}" \
		"${outtype}" \
		"${quant_threads}"
	touch "${stamp_path}"
}

kv_calib_extra=()
if [[ "${kv_calib_n_gpu_layers}" != "0" ]]; then
	kv_calib_extra+=(--n-gpu-layers "${kv_calib_n_gpu_layers}")
fi
if [[ -n "${kv_calib_device}" ]]; then
	kv_calib_extra+=(--device "${kv_calib_device}")
fi

cmake -S "${repo_root}" -B "${repo_root}/build"
cmake --build "${repo_root}/build" \
	--target llama-spectral-calibrate llama-spectral-kv-calibrate llama-quantize llama-cli llama-bench llama-perplexity \
	-j8

recipe_sq3_tmp="$(mktemp "${TMPDIR:-/tmp}/gemma4-sq3-expanded.XXXXXX")"
recipe_sq4_tmp="$(mktemp "${TMPDIR:-/tmp}/gemma4-sq4-expanded.XXXXXX")"
recipe_sq3_expanded="${recipe_sq3_tmp}.txt"
recipe_sq4_expanded="${recipe_sq4_tmp}.txt"
mv "${recipe_sq3_tmp}" "${recipe_sq3_expanded}"
mv "${recipe_sq4_tmp}" "${recipe_sq4_expanded}"
trap 'rm -f "${recipe_sq3_expanded}" "${recipe_sq4_expanded}"' EXIT

build_expanded_recipe "${recipe_sq3}" "${recipe_sq3_expanded}"
build_expanded_recipe "${recipe_sq4}" "${recipe_sq4_expanded}"

if [[ ! -f "${spectral_weights}" ]]; then
	spectral_match_regex="$(build_match_regex)"
	"${repo_root}/build/bin/llama-spectral-calibrate" \
		--input "${input_q8}" \
		--output "${spectral_weights}" \
		--match "${spectral_match_regex}" \
		--kind weight \
		--max-rows "${spectral_max_rows}" \
		--threads "${spectral_threads}" \
		--profile "${spectral_profile_set}"
fi

run_sq_quantize "${output_sq4}" "${recipe_sq4_expanded}" SQ4_1S
run_sq_quantize "${output_sq3}" "${recipe_sq3_expanded}" SQ3_1S

if [[ -z "${kv_calib_source}" && -f "${spectral_kv_gpu}" ]]; then
	kv_calib_source="${spectral_kv_gpu}"
fi

if [[ ! -f "${spectral_kv}" || "${force_rebuild}" == "1" ]]; then
	if [[ -n "${kv_calib_source}" ]]; then
		ln -f "${kv_calib_source}" "${spectral_kv}"
	else
		rm -f "${spectral_kv}"
		"${repo_root}/build/bin/llama-spectral-kv-calibrate" \
			--model "${input_q8}" \
			--output "${spectral_kv}" \
			--prompt-file "${bench_prompt_short}" \
			--prompt-file "${bench_prompt_full}" \
			--ctx-size 8192 \
			--batch-size 512 \
			--ubatch-size 512 \
			"${kv_calib_extra[@]}"
	fi
else
	printf 'reusing existing %s\n' "${spectral_kv}"
fi

"${repo_root}/scripts/sync-gemma4-chat-templates.sh"
"${repo_root}/scripts/repack-gemma4-chat-metadata.sh"

ln -f "${spectral_kv}" \
	"${repo_root}/models/hf-upload-gemma4-sq4/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf"

if [[ -f "${upload_readme}" ]]; then
	render_upload_readme
fi

printf 'built and repacked:\n'
printf '  %s\n' "${output_sq4}"
printf '  %s\n' "${output_sq3}"
printf '  %s\n' "${spectral_kv}"
