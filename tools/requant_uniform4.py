"""Rewrite a W4A8 checkpoint onto the uniform 4-bit grid.

`asym_w4a8_int8` checkpoints store a 16-level Lloyd-Max codebook, and the decoded
int8 levels differ per group, so a fast decode kernel would need one table lookup
per weight. This tool decodes each packed weight through the codebook, re-quantizes
it onto a uniform grid with a per-group scale search, and drops `weight_codebook`
so `w4a8_gemv.py` can unpack nibbles in registers.

Quality on real Qwen3.5-9B layers (relL2 against the int8 weight): codebook
0.0704, uniform + scale search 0.0750, uniform with a plain amax scale 0.0852.

    python tools/requant_uniform4.py --src in.safetensors --dst out.safetensors
"""

import argparse
import json
import os
import struct
import sys

import torch

DTYPE_SIZE = {"F32": 4, "BF16": 2, "F16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1,
              "F8_E5M2": 1, "I32": 4, "I64": 8, "U32": 4, "U64": 8, "BOOL": 1}
SCALE_CANDIDATES = (0.85, 0.95, 1.0, 1.05, 1.15)
ALS_ITERS = 2


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_size(entry):
    numel = 1
    for d in entry["shape"]:
        numel *= d
    return numel * DTYPE_SIZE[entry["dtype"]]


def search_scale(w, gs):
    """Per-group scale for a uniform signed 4-bit grid (codes -8..7)."""
    n, k = w.shape
    g = w.view(n, k // gs, gs)
    base = (g.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
    best_s, best_err = base, None
    for f in SCALE_CANDIDATES:
        s = base * f
        err = ((g / s).round().clamp(-8, 7) * s - g).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_s, best_err = s, err
        else:
            better = err < best_err
            best_s = torch.where(better, s, best_s)
            best_err = torch.where(better, err, best_err)
    for _ in range(ALS_ITERS):
        lv = (g / best_s).round().clamp(-8, 7)
        best_s = ((g * lv).sum(-1, keepdim=True)
                  / (lv * lv).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
    return best_s


def requant(qdata, s_rel, s_channel, codebook, gs):
    """Returns (packed nibbles, fp8 group scale) on the uniform grid."""
    from comfy_kitchen.backends.eager.w4a8_int8 import _dequant_int4_grouped_to_int8

    n = qdata.shape[0]
    k = s_rel.shape[1] * gs
    int8_w = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, gs)
    w = int8_w.float().view(n, k // gs, gs) * s_channel.view(-1, 1, 1)
    del int8_w

    s_rel_new = (search_scale(w.view(n, k), gs) / s_channel.view(-1, 1, 1)).to(torch.float8_e4m3fn)
    s_eff = s_rel_new.float() * s_channel.view(-1, 1, 1)
    codes = ((w / s_eff).round().clamp(-8, 7) + 8).to(torch.int32).view(n, k)
    packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    return packed, s_rel_new.view(n, k // gs).contiguous()


def _raw_bytes(t):
    t = t.contiguous().cpu()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.view(torch.uint8)
    return t.numpy().tobytes()


def write_safetensors(path, entries, src_path, header, data_start):
    header_out = {}
    off = 0
    for name, dtype, shape, size, _ in entries:
        header_out[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(header_out, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as out, open(src_path, "rb") as src:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, _, _, size, how in entries:
            if how[0] == "copy":
                e = header[how[1]]
                src.seek(data_start + e["data_offsets"][0])
                left = size
                while left:
                    chunk = src.read(min(left, 1 << 23))
                    if not chunk:
                        raise IOError(f"short read on {how[1]}")
                    out.write(chunk)
                    left -= len(chunk)
            else:
                buf = _raw_bytes(how[1])
                if len(buf) != size:
                    raise IOError(f"{name}: wrote {len(buf)} bytes, expected {size}")
                out.write(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N layers (smoke test)")
    args = ap.parse_args()
    if os.path.exists(args.dst):
        sys.exit(f"{args.dst} exists, refusing to overwrite")

    from safetensors import safe_open

    device = "cuda" if torch.cuda.is_available() else "cpu"
    header, data_start = read_header(args.src)

    layers = {}
    with safe_open(args.src, "pt") as f:
        for name in header:
            if not name.endswith(".weight_s_rel"):
                continue
            prefix = name[: -len("weight_s_rel")]
            if prefix + "weight_correction" in header:
                print(f"  skip asymmetric layer {prefix}")
                continue
            conf = f.get_tensor(prefix + "comfy_quant")
            meta = json.loads(bytes(conf.tolist()))
            if meta.get("format") == "asym_w4a8_int8":
                layers[prefix] = int(meta.get("group_size", 16))
    print(f"{len(layers)} grouped 4-bit layers in {os.path.basename(args.src)}")

    new_tensors = {}
    drop = set()
    total = min(args.limit, len(layers)) if args.limit else len(layers)
    with safe_open(args.src, "pt") as f:
        for i, (prefix, gs) in enumerate(sorted(layers.items())):
            if args.limit and i >= args.limit:
                break
            qdata = f.get_tensor(prefix + "weight").to(device)
            s_rel = f.get_tensor(prefix + "weight_s_rel").to(device)
            if s_rel.dtype == torch.uint8:
                s_rel = s_rel.view(torch.float8_e4m3fn)
            s_channel = f.get_tensor(prefix + "weight_s_channel").to(device).float()
            cb = f.get_tensor(prefix + "weight_codebook").to(device)
            packed, s_rel_new = requant(qdata, s_rel, s_channel, cb, gs)
            new_tensors[prefix + "weight"] = packed
            new_tensors[prefix + "weight_s_rel"] = s_rel_new
            drop.add(prefix + "weight_codebook")
            del qdata, s_rel, s_channel, cb
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{total} {prefix}")

    entries = []
    for name, e in header.items():
        if name == "__metadata__" or name in drop:
            continue
        t = new_tensors.get(name)
        if t is None:
            entries.append((name, e["dtype"], e["shape"], tensor_size(e), ("copy", name)))
        else:
            entries.append((name, e["dtype"], list(t.shape),
                            int(t.numel()) * DTYPE_SIZE[e["dtype"]], ("tensor", t)))

    print(f"writing {args.dst}")
    write_safetensors(args.dst, entries, args.src, header, data_start)
    print(f"done: {len(new_tensors) // 2} layers requantized, {os.path.getsize(args.dst) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
