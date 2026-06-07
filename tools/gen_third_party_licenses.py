#!/usr/bin/env python3
"""Generate THIRD-PARTY-LICENSES.md by harvesting LICENSE files from the cargo
registry cache (already on disk — no network). Dependency-free equivalent of
cargo-bundle-licenses.

Run from anywhere:  python3 tools/gen_third_party_licenses.py
Regenerate whenever Cargo.lock changes.
"""
import json, os, re, subprocess, glob, hashlib

# repo root = parent of this script's tools/ dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

meta = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--all-features", "--quiet"],
    cwd=ROOT))
ws = set(meta.get("workspace_members", []))

LIC_GLOBS = ["LICENSE*", "LICENCE*", "COPYING*", "COPYRIGHT*", "NOTICE*",
             "UNLICENSE*", "license*", "licence*"]

def is_text(p):
    if not os.path.isfile(p):
        return False
    if re.search(r"\.(rs|toml|md5|sha\d*|png|jpg|ico)$", p):
        return False
    return True

pkgs, texts, no_file = [], {}, {}

for p in meta["packages"]:
    if p["id"] in ws:
        continue
    name, ver = p["name"], p["version"]
    spdx = p.get("license") or "UNKNOWN"
    url = p.get("repository") or p.get("homepage") or ""
    pkgs.append((name, ver, spdx, url))
    d = os.path.dirname(p["manifest_path"])
    found = sorted({f for g in LIC_GLOBS
                    for f in glob.glob(os.path.join(d, g)) if is_text(f)})
    if not found:
        no_file.setdefault(spdx, set()).add(f"{name} {ver}")
        continue
    for f in found:
        try:
            raw = open(f, encoding="utf-8", errors="replace").read().strip()
        except Exception:
            continue
        if not raw:
            continue
        norm = re.sub(r"\s+", " ", raw).strip().lower()
        key = hashlib.sha256(norm.encode()).hexdigest()
        texts.setdefault(key, {"raw": raw, "crates": set()})
        texts[key]["crates"].add(f"{name} {ver}")

pkgs.sort(key=lambda x: x[0].lower())

out = []
out.append("# Third-Party Licenses\n")
out.append("`llmtrim` is licensed **AGPL-3.0-only**. It links the following third-party "
           "crates, each under its own license (all permissive or AGPL-compatible "
           "weak-copyleft). Their copyright notices and license texts are reproduced "
           "below to satisfy attribution requirements.\n")
out.append(f"Generated from `cargo metadata --all-features` — "
           f"{len(pkgs)} dependencies, {len(texts)} distinct license texts.\n")
out.append("> Regenerate after dependency changes: "
           "`python3 tools/gen_third_party_licenses.py`.\n")
out.append("\n---\n\n## Dependencies\n")
out.append("| Crate | Version | License (SPDX) |")
out.append("|-------|---------|----------------|")
for name, ver, spdx, url in pkgs:
    cell = f"[{name}]({url})" if url else name
    out.append(f"| {cell} | {ver} | `{spdx}` |")

if no_file:
    out.append("\n### Crates without an embedded license file\n")
    out.append("These declare an SPDX license but ship no `LICENSE` file in their "
               "package; the canonical text of the named license applies.\n")
    for spdx in sorted(no_file):
        out.append(f"- **`{spdx}`** — " + ", ".join(sorted(no_file[spdx])))

out.append("\n---\n\n## License Texts\n")
for key, v in sorted(texts.items(), key=lambda kv: (-len(kv[1]["crates"]),
                                                     sorted(kv[1]["crates"])[0].lower())):
    crates = sorted(v["crates"], key=str.lower)
    shown = ", ".join(crates[:40])
    if len(crates) > 40:
        shown += f", … (+{len(crates)-40} more)"
    out.append(f"\n<details>\n<summary><strong>Used by {len(crates)} crate(s):</strong> "
               f"{shown}</summary>\n\n```")
    out.append(v["raw"])
    out.append("```\n</details>\n")

dest = os.path.join(ROOT, "THIRD-PARTY-LICENSES.md")
open(dest, "w").write("\n".join(out))
print(f"wrote {dest}: {len(pkgs)} deps, {len(texts)} license texts, "
      f"{sum(len(s) for s in no_file.values())} crates without embedded file")
