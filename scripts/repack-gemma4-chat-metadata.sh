#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${PYTHON:-python3}"
gguf_tool="${repo_root}/scripts/gguf_repack_metadata_stream.py"
canonical_dir="${repo_root}/models/gemma4-96e-a4b-heretic"
repack_only="${REPACK_ONLY:-all}"

if [[ ! -f "${canonical_dir}/chat_template.jinja" ]]; then
	printf 'missing %s\n' "${canonical_dir}/chat_template.jinja" >&2
	exit 1
fi

config_tmp="$(mktemp "${TMPDIR:-/tmp}/gemma4-chat-templates.XXXXXX")"
config_json="${config_tmp}.json"
mv "${config_tmp}" "${config_json}"
cleanup() {
	rm -f "${config_json}"
}
trap cleanup EXIT

mkdir -p \
	"${repo_root}/models/hf-upload-gemma4-tq3" \
	"${repo_root}/models/hf-upload-gemma4-tq4" \
	"${repo_root}/models/hf-upload-gemma4-sq3" \
	"${repo_root}/models/hf-upload-gemma4-sq4"

"${python_bin}" - <<'PY' "${canonical_dir}" "${config_json}"
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])

config = {
    "chat_template": [
        {
            "name": "default",
            "template": (model_dir / "chat_template.jinja").read_text(encoding="utf-8"),
        },
        {
            "name": "standard",
            "template": (model_dir / "additional_chat_templates" / "standard.jinja").read_text(encoding="utf-8"),
        },
    ]
}

config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
PY

repack_one() {
	local input="$1"
	local tmp_output="${input}.tmp-chatmeta"

	rm -f "${tmp_output}"
	"${python_bin}" "${gguf_tool}" "${input}" "${tmp_output}" \
		--chat-template-config "${config_json}" \
		--force
	mv "${tmp_output}" "${input}"
}

if [[ "${repack_only}" == "all" || "${repack_only}" == "tq" ]]; then
	repack_one "${canonical_dir}/Gemma-4-96E-A4B-Heretic-TQ4_1S.gguf"
	repack_one "${canonical_dir}/Gemma-4-96E-A4B-Heretic-TQ3_1S.gguf"
fi
if [[ "${repack_only}" == "all" || "${repack_only}" == "sq" ]]; then
	if [[ -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf" ]]; then
		repack_one "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf"
	fi
	if [[ -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf" ]]; then
		repack_one "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
	fi
fi

if [[ "${repack_only}" == "all" || "${repack_only}" == "tq" ]]; then
	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-TQ4_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-tq4/Gemma-4-96E-A4B-Heretic-TQ4_1S.gguf"

	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-TQ3_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-tq4/Gemma-4-96E-A4B-Heretic-TQ3_1S.gguf"

	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-TQ3_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-tq3/Gemma-4-96E-A4B-Heretic-TQ3_1S.gguf"
fi

if [[ -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf" ]]; then
	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-sq4/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf"
fi

if [[ -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf" ]]; then
	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-sq4/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
	ln -f "${canonical_dir}/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf" \
		"${repo_root}/models/hf-upload-gemma4-sq3/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf"
fi

printf 'repacked Gemma 4 TQ/SQ GGUF metadata with default=interleaved, named=standard\n'
