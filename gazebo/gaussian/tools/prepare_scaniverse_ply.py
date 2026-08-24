#!/usr/bin/env python3
"""Remove non-finite Scaniverse Gaussian records without changing the source."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import tempfile


def read_header(stream) -> tuple[list[bytes], int, int]:
    lines: list[bytes] = []
    vertex_count = 0
    float_properties = 0
    in_vertex = False

    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY ended before end_header")
        lines.append(line)
        text = line.decode("ascii").strip()
        if text == "format binary_little_endian 1.0":
            continue
        if text.startswith("element "):
            fields = text.split()
            in_vertex = fields[1] == "vertex"
            if in_vertex:
                vertex_count = int(fields[2])
            continue
        if in_vertex and text.startswith("property float "):
            float_properties += 1
        if text == "end_header":
            break

    if vertex_count <= 0 or float_properties <= 0:
        raise ValueError("PLY has no floating-point Gaussian vertices")
    return lines, vertex_count, float_properties


def rewritten_header(lines: list[bytes], vertex_count: int) -> bytes:
    output: list[bytes] = []
    for line in lines:
        if line.startswith(b"element vertex "):
            output.append(f"element vertex {vertex_count}\n".encode("ascii"))
        else:
            output.append(line)
    return b"".join(output)


def prepare(source: Path, destination: Path, max_position: float | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src:
        header, input_count, property_count = read_header(src)
        record_size = property_count * 4
        unpacker = struct.Struct(f"<{property_count}f")
        retained: list[bytes] = []
        nonfinite = 0
        distant = 0

        for _ in range(input_count):
            record = src.read(record_size)
            if len(record) != record_size:
                raise ValueError("PLY vertex data is truncated")
            values = unpacker.unpack(record)
            if not all(math.isfinite(value) for value in values):
                nonfinite += 1
                continue
            if max_position is not None and max(abs(value) for value in values[:3]) > max_position:
                distant += 1
                continue
            retained.append(record)

        if src.read(1):
            raise ValueError("unexpected data follows the Gaussian vertex block")

    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(rewritten_header(header, len(retained)))
        for record in retained:
            tmp.write(record)
    tmp_path.replace(destination)

    print(f"Input Gaussians:    {input_count}")
    print(f"Removed nonfinite:  {nonfinite}")
    print(f"Removed outliers:   {distant}")
    print(f"Output Gaussians:   {len(retained)}")
    print(f"Output:             {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--max-position",
        type=float,
        default=10.0,
        help="drop records outside +/- this many source units; <=0 disables it",
    )
    args = parser.parse_args()
    prepare(
        args.source.resolve(),
        args.destination.resolve(),
        args.max_position if args.max_position > 0 else None,
    )


if __name__ == "__main__":
    main()
