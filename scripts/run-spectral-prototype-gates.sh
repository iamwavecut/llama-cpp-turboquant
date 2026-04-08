#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR="build"
RUN_ACCEPTANCE=0
REBUILD=0

usage() {
	cat <<'EOF'
Usage: scripts/run-spectral-prototype-gates.sh [options]

Options:
  --build-dir <dir>      CMake build directory (default: build)
  --with-acceptance      Run the tiny real-GGUF acceptance gate in addition to synthetic tests
  --rebuild              Rebuild the required test targets before running
  -h, --help             Show this help

Default behavior runs only the fast synthetic SpectralQuant prototype gates:
  - test-spectral-quant
  - test-spectral-kv-state

Add --with-acceptance to also run:
  - test-download-model
  - test-spectral-acceptance
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--build-dir)
			BUILD_DIR="${2:?missing value for --build-dir}"
			shift 2
			;;
		--with-acceptance)
			RUN_ACCEPTANCE=1
			shift
			;;
		--rebuild)
			REBUILD=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "error: unknown argument: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

if [[ ! -d "$BUILD_DIR" ]]; then
	echo "error: build directory not found: $BUILD_DIR" >&2
	exit 1
fi

run_step() {
	local title="$1"
	shift
	printf '\n[%s]\n' "$title"
	"$@"
}

if [[ "$REBUILD" -eq 1 ]]; then
	run_step "build synthetic spectral tests" \
		cmake --build "$BUILD_DIR" --target test-spectral-quant test-spectral-kv-state -j8

	if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
		run_step "build tiny-fixture acceptance test" \
			cmake --build "$BUILD_DIR" --target test-spectral-acceptance -j8
	fi
fi

run_step "synthetic spectral gates" \
	ctest --test-dir "$BUILD_DIR" -R 'test-spectral-(quant|kv-state)' --output-on-failure

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
	run_step "tiny real-gguf spectral gate" \
		ctest --test-dir "$BUILD_DIR" -R 'test-download-model|test-spectral-acceptance' --output-on-failure
fi

printf '\n[ok] spectral prototype gates passed\n'
