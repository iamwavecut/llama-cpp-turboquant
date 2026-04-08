#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

standard_template="${repo_root}/models/templates/google-gemma-4-31B-it.jinja"
interleaved_template="${repo_root}/models/templates/google-gemma-4-31B-it-interleaved.jinja"
canonical_recipes_dir="${repo_root}/models/gemma4-96e-a4b-heretic"
bench_prompt_short="${canonical_recipes_dir}/bench_prompt_short.txt"
bench_prompt_full="${canonical_recipes_dir}/bench_prompt_full.txt"

targets=(
	"${repo_root}/models/gemma4-96e-a4b-heretic"
	"${repo_root}/models/hf-upload-gemma4-tq3"
	"${repo_root}/models/hf-upload-gemma4-tq4"
	"${repo_root}/models/hf-upload-gemma4-sq3"
	"${repo_root}/models/hf-upload-gemma4-sq4"
)

for dir in "${targets[@]}"; do
	if [[ ! -d "${dir}" ]]; then
		continue
	fi

	mkdir -p "${dir}/additional_chat_templates"
	cp "${interleaved_template}" "${dir}/chat_template.jinja"
	cp "${standard_template}" "${dir}/additional_chat_templates/standard.jinja"
	rm -f "${dir}/additional_chat_templates/interleaved.jinja"

	if [[ "${dir}" == "${canonical_recipes_dir}" ]]; then
		if [[ -f "${dir}/config_i_gemma4_30l.txt" ]]; then
			mv "${dir}/config_i_gemma4_30l.txt" "${dir}/requant_recipe_tq4_1s.txt"
		fi
		if [[ -f "${dir}/config_i_gemma4_30l_tq3.txt" ]]; then
			mv "${dir}/config_i_gemma4_30l_tq3.txt" "${dir}/requant_recipe_tq3_1s.txt"
		fi
	else
		if [[ "${dir}" == "${repo_root}/models/hf-upload-gemma4-tq3" || "${dir}" == "${repo_root}/models/hf-upload-gemma4-tq4" ]]; then
			if [[ -f "${canonical_recipes_dir}/requant_recipe_tq4_1s.txt" && "${dir}" != "${repo_root}/models/hf-upload-gemma4-tq3" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_tq4_1s.txt" "${dir}/requant_recipe_tq4_1s.txt"
			fi
			if [[ -f "${canonical_recipes_dir}/requant_recipe_tq3_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_tq3_1s.txt" "${dir}/requant_recipe_tq3_1s.txt"
			fi
			rm -f "${dir}/requant_recipe_sq3_1s.txt" "${dir}/requant_recipe_sq4_1s.txt"
			rm -f "${dir}/bench_prompt_short.txt" "${dir}/bench_prompt_full.txt"
		elif [[ "${dir}" == "${repo_root}/models/hf-upload-gemma4-sq3" || "${dir}" == "${repo_root}/models/hf-upload-gemma4-sq4" ]]; then
			if [[ -f "${canonical_recipes_dir}/requant_recipe_sq4_1s.txt" && "${dir}" != "${repo_root}/models/hf-upload-gemma4-sq3" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_sq4_1s.txt" "${dir}/requant_recipe_sq4_1s.txt"
			fi
			if [[ -f "${canonical_recipes_dir}/requant_recipe_sq3_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_sq3_1s.txt" "${dir}/requant_recipe_sq3_1s.txt"
			fi
			if [[ -f "${bench_prompt_short}" ]]; then
				cp "${bench_prompt_short}" "${dir}/bench_prompt_short.txt"
			fi
			if [[ -f "${bench_prompt_full}" ]]; then
				cp "${bench_prompt_full}" "${dir}/bench_prompt_full.txt"
			fi
			rm -f "${dir}/requant_recipe_tq3_1s.txt" "${dir}/requant_recipe_tq4_1s.txt"
		else
			if [[ -f "${canonical_recipes_dir}/requant_recipe_tq4_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_tq4_1s.txt" "${dir}/requant_recipe_tq4_1s.txt"
			fi
			if [[ -f "${canonical_recipes_dir}/requant_recipe_tq3_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_tq3_1s.txt" "${dir}/requant_recipe_tq3_1s.txt"
			fi
			if [[ -f "${canonical_recipes_dir}/requant_recipe_sq4_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_sq4_1s.txt" "${dir}/requant_recipe_sq4_1s.txt"
			fi
			if [[ -f "${canonical_recipes_dir}/requant_recipe_sq3_1s.txt" ]]; then
				cp "${canonical_recipes_dir}/requant_recipe_sq3_1s.txt" "${dir}/requant_recipe_sq3_1s.txt"
			fi
		fi
		rm -f "${dir}/config_i_gemma4_30l.txt" "${dir}/config_i_gemma4_30l_tq3.txt"
	fi

	printf 'synced %s\n' "${dir#"${repo_root}/"}"
done
