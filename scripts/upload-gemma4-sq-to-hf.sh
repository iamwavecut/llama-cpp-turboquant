#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
upload_dir="${repo_root}/models/hf-upload-gemma4-sq4"
repo_id="WaveCut/Gemma-4-96E-A4B-Heretic-SQ"

if [[ ! -d "${upload_dir}" ]]; then
	printf 'missing upload directory: %s\n' "${upload_dir}" >&2
	exit 1
fi

required_files=(
	"${upload_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf"
	"${upload_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
	"${upload_dir}/Gemma-4-96E-A4B-Heretic-SKV3-calibration.gguf"
	"${upload_dir}/README.md"
)

for path in "${required_files[@]}"; do
	if [[ ! -f "${path}" ]]; then
		printf 'missing release artifact: %s\n' "${path}" >&2
		exit 1
	fi
done

if rg -q 'SIZE_(SQ4|SQ3|SKV3)_(BYTES|GIB)' "${upload_dir}/README.md"; then
	printf 'README still contains unresolved size placeholders: %s\n' "${upload_dir}/README.md" >&2
	exit 1
fi

hf repo create "${repo_id}" --repo-type model --exist-ok
hf upload-large-folder "${repo_id}" "${upload_dir}" --repo-type model
