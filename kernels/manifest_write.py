"""Surgical edits of a manifest file: change lines, keep the comments.

A bench run must not reformat a hand-written file -- the comments explain why a
number is what it is, and they are the reason the table can be reviewed at all.
So this module edits the TOML *text* instead of re-serialising a parsed tree.
Only two things are ever touched:

    [arch] fingerprint = "..."
    [sites."<name>".shapes] -> the keys a run actually measured

Everything else (other sites, notes, ordering) is preserved byte for byte, and
the result is parsed again before it is written: a generated manifest that does
not load would break every future start, so `save()` refuses.
"""

from pathlib import Path

from .manifest import SCHEMA_VERSION, _loads


def _block_range(lines, header):
    """(first line after `header`, index of the next header) or None."""
    for index, line in enumerate(lines):
        if line.strip() == header:
            end = index + 1
            while end < len(lines) and not lines[end].lstrip().startswith("["):
                end += 1
            return index, end
    return None


def _line_key(line):
    """The TOML key a `key = value` line defines, unquoted, or None."""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    key, sep, _ = text.partition("=")
    if not sep:
        return None
    key = key.strip()
    if len(key) >= 2 and key[0] == '"' and key[-1] == '"':
        key = key[1:-1]
    return key


def _join(lines):
    return "\n".join(lines).rstrip("\n") + "\n"


def skeleton(arch, fingerprint):
    """A minimal manifest for a machine that has no file yet."""
    return (f"schema = {SCHEMA_VERSION}\n\n"
            f"[arch]\n"
            f'gcn = "{arch}"\n'
            f'fingerprint = "{fingerprint}"\n\n'
            f"# Written by tools/bench.py: each shape lists the implementations that\n"
            f"# were timed on this machine, plus the numeric error against tools/refs.py.\n")


def set_arch_identity(text, arch, fingerprint):
    """Stamp the ROCm/Triton/gcn triplet this file's numbers were measured under."""
    lines = text.splitlines()
    rng = _block_range(lines, "[arch]")
    line = f'fingerprint = "{fingerprint}"'
    if rng is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", "[arch]", f'gcn = "{arch}"', line, ""]
        return _join(lines)
    start, end = rng
    for index in range(start + 1, end):
        if lines[index].strip().startswith("fingerprint"):
            lines[index] = line
            return _join(lines)
    lines.insert(start + 1, line)
    return _join(lines)


def set_shapes(text, site, records):
    """Replace/add shape lines in `[sites."<site>".shapes]`.

    `records` maps a shape key ("4096x4096@m1") to its rendered line. Keys that
    are not in `records` are left untouched, so a run that only timed three
    shapes does not drop the rest of the table.
    """
    header = f'[sites."{site}".shapes]'
    lines = text.splitlines()
    rng = _block_range(lines, header)
    if rng is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", header] + list(records.values()) + [""]
        return _join(lines)

    start, end = rng
    body = lines[start + 1:end]
    pending = dict(records)
    for index, line in enumerate(body):
        key = _line_key(line)
        if key is not None and key in pending:
            body[index] = pending.pop(key)
    while body and not body[-1].strip():
        body.pop()
    for key, line in records.items():
        if key in pending:
            body.append(line)
    lines[start + 1:end] = body + [""]
    return _join(lines)


def set_default(text, site, impl, grade="measured"):
    """Set the site-wide fallback used for shapes the table does not cover.

    Only called when every measured shape of the site agrees on the winner:
    a default is a generalisation, so it should be backed by the whole row of
    numbers, not by one lucky shape.
    """
    header = f'[sites."{site}"]'
    want_default = f'default = "{impl}"'
    want_grade = f'default_grade = "{grade}"'
    lines = text.splitlines()
    rng = _block_range(lines, header)

    if rng is None:
        block = ["", header, want_default, want_grade, ""]
        shapes_header = f'[sites."{site}".shapes]'
        index = next((i for i, line in enumerate(lines) if line.strip() == shapes_header), None)
        if index is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += block
        else:
            lines[index:index] = block
        return _join(lines)

    start, end = rng
    body, seen = [], set()
    for line in lines[start + 1:end]:
        key = _line_key(line)
        if key == "default":
            body.append(want_default)
            seen.add(key)
        elif key == "default_grade":
            body.append(want_grade)
            seen.add(key)
        elif line.strip():
            body.append(line)
    if "default" not in seen:
        body.insert(0, want_default)
    if "default_grade" not in seen:
        body.insert(1, want_grade)
    lines[start + 1:end] = body + [""]
    return _join(lines)


def _num(value):
    """A TOML float that stays readable at the precision we measure."""
    return f"{value:.6g}"


def render(key, impl, ms, rel=None, grade="measured", date="", note=""):
    """One shape line: winner, every timed implementation, and the error."""
    times = ", ".join(f"{name} = {_num(value)}" for name, value in sorted(ms.items()))
    parts = [f'impl = "{impl}"', f"ms = {{ {times} }}"]
    if rel is not None:
        parts.append(f"rel = {rel:.2e}")
    parts.append(f'grade = "{grade}"')
    if date:
        parts.append(f'date = "{date}"')
    if note:
        parts.append(f'note = "{note}"')
    return f'"{key}" = {{ {", ".join(parts)} }}'


def save(path, text):
    """Parse-then-write. Refuses to leave a manifest that cannot be loaded."""
    try:
        _loads(text)
    except Exception as exc:                        # noqa: BLE001 - the point is to refuse
        raise ValueError(f"generated manifest for {path} does not parse ({exc}); nothing written") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
