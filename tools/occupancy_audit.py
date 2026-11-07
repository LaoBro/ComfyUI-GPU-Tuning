"""Audit occupancy of hand-written HIP kernels from their .so, no GPU needed.

    python tools/occupancy_audit.py                 # all arch/gfx906/kernels/*.so
    python tools/occupancy_audit.py a.so b.so --asm # specific libs + dot classification

Why this exists: int8_gemm.cu was spill-free at 211 VGPR and looked healthy, but
211 > 128 silently pinned it to 1 block/CU (4 waves) while its 16 KB LDS would
have allowed 4. A competitor kept the same 64-accumulator tile at 122 VGPR (2
blocks/CU, 8 waves) and was 1.6x faster. spill_count alone does not catch this;
you have to read vgpr_count against the occupancy threshold. This tool does that
for every kernel in a lib and flags the register-bound ones.

gfx906 model (see docs/HIP_KERNELS.md): 64 lanes/wave, 4 SIMD/CU, 16384 VGPR
dwords/SIMD (allocated in granularity 4), 64 KB LDS/CU, <=10 waves/SIMD, <=16
blocks/CU. Blocks are assumed to spread evenly across the 4 SIMDs, which is exact
for the 256-thread (4-wave) blocks these kernels use.
"""

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
EM_AMDGPU = 224
VGPR_PER_SIMD = 16384
LANES = 64
SIMDS = 4
LDS_PER_CU = 65536
MAX_WAVES_SIMD = 10
MAX_BLOCKS_CU = 16

# Theoretical dot peaks, TMAC/s (docs/HIP_KERNELS.md hardware table). fma_f32 is
# 1 MAC/instr; the dots are 2/4/8. pk_fma_f16 is 2 MAC but a lower issue rate.
PEAK_TMAC = {"dot2_f16": 13.4, "dot4_i8": 26.5, "dot8_i4": 53.0,
             "fma_f32": 6.62, "pk_fma_f16": 13.4, "?": 0.0}
DOT_TYPES = {"dot2_f16", "dot4_i8", "dot8_i4"}


def device_elfs(so_path):
    """Yield (offset, blob) for each AMDGPU code object embedded in the fatbin."""
    data = Path(so_path).read_bytes()
    for m in re.finditer(b"\x7fELF", data):
        off = m.start()
        hdr = data[off:off + 64]
        if len(hdr) < 64 or hdr[4] != 2:           # ELF64 only
            continue
        if struct.unpack_from("<H", hdr, 18)[0] != EM_AMDGPU:
            continue
        e_shoff = struct.unpack_from("<Q", hdr, 40)[0]
        e_shentsize = struct.unpack_from("<H", hdr, 58)[0]
        e_shnum = struct.unpack_from("<H", hdr, 60)[0]
        end = off + e_shoff + e_shentsize * e_shnum
        if end > len(data):
            continue
        yield off, data[off:end]


def read_notes(blob):
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        out = subprocess.run(["llvm-readelf", "--notes", path],
                             capture_output=True, text=True).stdout
    finally:
        Path(path).unlink()
    kernels = []
    for blk in out.split("- .args:"):
        def field(name):
            m = re.search(rf"\.{name}:\s*(\S+)", blk)
            return m.group(1) if m else None
        name = field("name")
        if not name:
            continue
        kernels.append({
            "name": name,
            "vgpr": int(field("vgpr_count") or 0),
            "spill": int(field("vgpr_spill_count") or 0),
            "sgpr": int(field("sgpr_count") or 0),
            "lds": int(field("group_segment_fixed_size") or 0),
            "wg": int(field("max_flat_workgroup_size") or 256),
            "wave": int(field("wavefront_size") or 64),
        })
    return kernels, path


def occupancy(k):
    vgpr = (k["vgpr"] + 3) & ~3                    # allocation granularity 4
    waves_per_block = max(1, k["wg"] // k["wave"])
    waves_simd = min(MAX_WAVES_SIMD, VGPR_PER_SIMD // (vgpr * LANES)) if vgpr else MAX_WAVES_SIMD
    vgpr_blocks = (waves_simd * SIMDS) // waves_per_block
    lds_blocks = LDS_PER_CU // k["lds"] if k["lds"] else MAX_BLOCKS_CU
    blocks = max(1, min(vgpr_blocks, lds_blocks, MAX_BLOCKS_CU))
    return {"waves_simd": waves_simd, "vgpr_blocks": vgpr_blocks,
            "lds_blocks": lds_blocks, "blocks": blocks,
            "waves_cu": blocks * waves_per_block, "waves_per_block": waves_per_block}


def classify_dot(blob, name):
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        out = subprocess.run(["llvm-objdump", "-d", f"--disassemble-symbols={name}", path],
                             capture_output=True, text=True).stdout
    finally:
        Path(path).unlink()
    counts = {"dot2_f16": out.count("v_dot2_f32_f16"),
              "dot4_i8": out.count("v_dot4_i32_i8"),
              "dot8_i4": out.count("v_dot8_i32_i4"),
              "pk_fma_f16": out.count("v_pk_fma_f16"),
              "fma_f32": len(re.findall(r"\bv_(?:fma|mac)_f32\b", out))}
    best = max(counts, key=counts.get)
    return (best if counts[best] else "?"), counts


def short(name):
    # strip the Itanium mangling noise for a readable column
    m = re.search(r"kernels\d+(\w+?)E", name)
    s = m.group(1) if m else name
    return s[:42]


def audit(so_path, want_asm):
    print(f"\n=== {Path(so_path).name} ===")
    print(f"{'kernel':42s} {'vgpr':>5s} {'spl':>4s} {'LDS':>6s} {'wg':>4s} "
          f"{'blk/CU':>6s} {'wav/CU':>6s} {'bound':>10s} {'dot':>10s} {'peak':>6s}")
    seen = set()
    flagged = []
    for off, blob in device_elfs(so_path):
        kernels, _ = read_notes(blob)
        for k in kernels:
            if k["name"] in seen:
                continue
            seen.add(k["name"])
            occ = occupancy(k)
            # Only actionable when a dot-issue-bound kernel is stuck BELOW the
            # ~8-wave ceiling knee by VGPR. A bandwidth-bound GEMV or the
            # butterfly rotate gains nothing from more waves, and a dot kernel
            # already at 8+ waves has saturated the ceiling — both are benign.
            reg_bound = occ["vgpr_blocks"] < occ["lds_blocks"] and k["spill"] == 0
            dot, peak = "?", 0.0
            if want_asm:
                dot, _ = classify_dot(blob, k["name"])
                peak = PEAK_TMAC[dot]
            actionable = reg_bound and dot in DOT_TYPES and occ["waves_cu"] < 8
            if actionable:
                tag = "REG-BOUND!"
                flagged.append((k["name"], k["vgpr"], occ, dot))
            elif reg_bound:
                tag = "reg-lim"
            elif k["spill"]:
                tag = "spill!"
            else:
                tag = ""
            print(f"{short(k['name']):42s} {k['vgpr']:5d} {k['spill']:4d} {k['lds']:6d} "
                  f"{k['wg']:4d} {occ['blocks']:6d} {occ['waves_cu']:6d} {tag:>10s} "
                  f"{dot:>10s} {peak:6.1f}")
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", nargs="*", help=".so files (default: arch/gfx906/kernels/*.so)")
    ap.add_argument("--asm", action="store_true", help="disassemble to classify the dot instruction")
    args = ap.parse_args()
    libs = args.so or sorted(str(p) for p in (NODE_DIR / "arch" / "gfx906" / "kernels").glob("*.so"))
    if not libs:
        print("no .so found", file=sys.stderr)
        return 1
    all_flagged = []
    for lib in libs:
        all_flagged += [(lib, *f) for f in audit(lib, args.asm)]
    print("\n=== actionable: dot-issue-bound kernels stuck below 8 waves by VGPR ===")
    if not all_flagged:
        print("none")
    for lib, name, vgpr, occ, dot in all_flagged:
        print(f"  {Path(lib).name}: {short(name)} [{dot}] vgpr={vgpr} -> {occ['blocks']} block/CU "
              f"({occ['waves_cu']} waves); LDS would allow {occ['lds_blocks']}; "
              f"peak {PEAK_TMAC[dot]:.1f} TMAC/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())