#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf

logger = logging.getLogger("gguf-repack-metadata-stream")


@dataclass(frozen=True)
class KVField:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedGGUF:
    endian_prefix: str
    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    kv_fields: list[KVField]
    tensor_info_start: int
    tensor_info_end: int
    data_offset: int


SCALAR_SIZES = {
    gguf.GGUFValueType.UINT8: 1,
    gguf.GGUFValueType.INT8: 1,
    gguf.GGUFValueType.UINT16: 2,
    gguf.GGUFValueType.INT16: 2,
    gguf.GGUFValueType.UINT32: 4,
    gguf.GGUFValueType.INT32: 4,
    gguf.GGUFValueType.FLOAT32: 4,
    gguf.GGUFValueType.BOOL: 1,
    gguf.GGUFValueType.UINT64: 8,
    gguf.GGUFValueType.INT64: 8,
    gguf.GGUFValueType.FLOAT64: 8,
}


def read_exact(fin: Any, size: int) -> bytes:
    data = fin.read(size)
    if len(data) != size:
        raise EOFError(f"unexpected EOF while reading {size} bytes")
    return data


def unpack_u32(fin: Any, endian: str) -> int:
    return struct.unpack(f"{endian}I", read_exact(fin, 4))[0]


def unpack_u64(fin: Any, endian: str) -> int:
    return struct.unpack(f"{endian}Q", read_exact(fin, 8))[0]


def read_string(fin: Any, endian: str) -> str:
    size = unpack_u64(fin, endian)
    return read_exact(fin, size).decode("utf-8")


def skip_string(fin: Any, endian: str) -> None:
    fin.seek(unpack_u64(fin, endian), os.SEEK_CUR)


def skip_value(fin: Any, value_type: gguf.GGUFValueType, endian: str) -> None:
    scalar_size = SCALAR_SIZES.get(value_type)
    if scalar_size is not None:
        fin.seek(scalar_size, os.SEEK_CUR)
        return

    if value_type == gguf.GGUFValueType.STRING:
        skip_string(fin, endian)
        return

    if value_type == gguf.GGUFValueType.ARRAY:
        subtype = gguf.GGUFValueType(unpack_u32(fin, endian))
        length = unpack_u64(fin, endian)
        scalar_size = SCALAR_SIZES.get(subtype)
        if scalar_size is not None:
            fin.seek(length * scalar_size, os.SEEK_CUR)
            return
        if subtype == gguf.GGUFValueType.STRING:
            for _ in range(length):
                skip_string(fin, endian)
            return
        for _ in range(length):
            skip_value(fin, subtype, endian)
        return

    raise ValueError(f"unsupported GGUF value type: {value_type}")


def parse_gguf(path: Path) -> ParsedGGUF:
    with path.open("rb") as fin:
        magic = struct.unpack("<I", read_exact(fin, 4))[0]
        if magic != gguf.GGUF_MAGIC:
            raise ValueError("GGUF magic invalid")

        raw_version = read_exact(fin, 4)
        version_le = struct.unpack("<I", raw_version)[0]
        endian = ">" if (version_le & 0xFFFF) == 0 else "<"
        version = struct.unpack(f"{endian}I", raw_version)[0]

        tensor_count = unpack_u64(fin, endian)
        kv_count = unpack_u64(fin, endian)
        alignment = gguf.GGUF_DEFAULT_ALIGNMENT
        kv_fields: list[KVField] = []

        for _ in range(kv_count):
            start = fin.tell()
            key = read_string(fin, endian)
            value_type = gguf.GGUFValueType(unpack_u32(fin, endian))

            if key == gguf.Keys.General.ALIGNMENT and value_type == gguf.GGUFValueType.UINT32:
                alignment = unpack_u32(fin, endian)
            else:
                skip_value(fin, value_type, endian)

            kv_fields.append(KVField(name=key, start=start, end=fin.tell()))

        tensor_info_start = fin.tell()
        for _ in range(tensor_count):
            skip_string(fin, endian)
            n_dims = unpack_u32(fin, endian)
            fin.seek(8 * n_dims + 4 + 8, os.SEEK_CUR)
        tensor_info_end = fin.tell()
        data_offset = gguf.GGUFWriter.ggml_pad(tensor_info_end, alignment)

    return ParsedGGUF(
        endian_prefix=endian,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        alignment=alignment,
        kv_fields=kv_fields,
        tensor_info_start=tensor_info_start,
        tensor_info_end=tensor_info_end,
        data_offset=data_offset,
    )


def serialize_chat_template_kv(endian_prefix: str, templates: Any) -> tuple[int, bytes]:
    endianess = gguf.GGUFEndian.BIG if endian_prefix == ">" else gguf.GGUFEndian.LITTLE
    writer = gguf.GGUFWriter(Path("/tmp/gguf-unused"), arch="unused", endianess=endianess)
    writer.add_chat_template(templates)

    kv_bytes = bytearray()
    kv_items = writer.kv_data[0]
    allowed_prefixes = (
        gguf.Keys.Tokenizer.CHAT_TEMPLATE,
        gguf.Keys.Tokenizer.CHAT_TEMPLATE_N.format(name=""),
        gguf.Keys.Tokenizer.CHAT_TEMPLATES,
    )

    kept_count = 0
    for key, val in kv_items.items():
        if not key.startswith(allowed_prefixes):
            continue
        kv_bytes += writer._pack_val(key, gguf.GGUFValueType.STRING, add_vtype=False)
        kv_bytes += writer._pack_val(val.value, val.type, add_vtype=True, sub_type=val.sub_type)
        kept_count += 1

    return kept_count, bytes(kv_bytes)


def copy_range(fin: Any, fout: Any, start: int, end: int) -> None:
    fin.seek(start)
    remaining = end - start
    while remaining > 0:
        chunk = fin.read(min(64 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("unexpected EOF while copying range")
        fout.write(chunk)
        remaining -= len(chunk)


def write_repacked_file(input_path: Path, output_path: Path, parsed: ParsedGGUF, chat_templates: Any) -> None:
    skip_prefixes = (
        gguf.Keys.Tokenizer.CHAT_TEMPLATE,
        gguf.Keys.Tokenizer.CHAT_TEMPLATE_N.format(name=""),
        gguf.Keys.Tokenizer.CHAT_TEMPLATES,
    )
    kept_fields: list[KVField] = []
    seen_keys: set[str] = set()
    for field in parsed.kv_fields:
        if field.name.startswith(skip_prefixes):
            continue
        if field.name in seen_keys:
            logger.warning("dropping duplicate GGUF metadata key during repack: %s", field.name)
            continue
        kept_fields.append(field)
        seen_keys.add(field.name)
    added_kv_count, added_kv_bytes = serialize_chat_template_kv(parsed.endian_prefix, chat_templates)
    kv_count = len(kept_fields) + added_kv_count
    payload_size = input_path.stat().st_size - parsed.data_offset

    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        fout.write(struct.pack("<I", gguf.GGUF_MAGIC))
        fout.write(struct.pack(f"{parsed.endian_prefix}I", parsed.version))
        fout.write(struct.pack(f"{parsed.endian_prefix}Q", parsed.tensor_count))
        fout.write(struct.pack(f"{parsed.endian_prefix}Q", kv_count))

        for field in kept_fields:
            copy_range(fin, fout, field.start, field.end)

        fout.write(added_kv_bytes)
        copy_range(fin, fout, parsed.tensor_info_start, parsed.tensor_info_end)

        pad = gguf.GGUFWriter.ggml_pad(fout.tell(), parsed.alignment) - fout.tell()
        if pad > 0:
            fout.write(b"\x00" * pad)

        fin.seek(parsed.data_offset)
        bar = tqdm(desc="Writing", total=payload_size, unit="byte", unit_scale=True)
        while True:
            chunk = fin.read(64 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            bar.update(len(chunk))
        bar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack GGUF metadata by raw-copying KV/tensor sections and replacing chat templates")
    parser.add_argument("input", type=Path, help="GGUF input filename")
    parser.add_argument("output", type=Path, help="GGUF output filename")
    parser.add_argument("--chat-template-config", type=Path, required=True, metavar="tokenizer_config.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    with args.chat_template_config.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    templates = config.get("chat_template")
    if not templates:
        raise SystemExit("chat template config does not contain chat_template")

    if args.output.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.output}")
        args.output.unlink()

    logger.info("* Loading: %s", args.input)
    parsed = parse_gguf(args.input)
    logger.info("* Writing: %s", args.output)
    write_repacked_file(args.input, args.output, parsed, templates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
