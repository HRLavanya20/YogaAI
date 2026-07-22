"""One-off script to restore pre-body-overlay files from agent transcript."""
import json
from pathlib import Path

TRANSCRIPT = Path(
    r"C:\Users\Greeshma\.cursor\projects\d-engg-major-prj-codebase-YogaAI"
    r"\agent-transcripts\b3d1e6ae-950e-473f-8b95-22002ebd1c6c"
    r"\b3d1e6ae-950e-473f-8b95-22002ebd1c6c.jsonl"
)
BASE = Path(__file__).resolve().parent
MAX_LINE = 79


def target_rel(path: str) -> str | None:
    p = path.replace("\\", "/").lower()
    if "ar_module/" in p:
        return "ar_module/" + p.split("ar_module/", 1)[1]
    return None


def load_transcript_ops():
    writes = {}
    patches = []
    for i, line in enumerate(TRANSCRIPT.read_text(encoding="utf-8").splitlines(), 1):
        if i > MAX_LINE:
            break
        obj = json.loads(line)
        for part in obj.get("message", {}).get("content", []):
            if part.get("type") != "tool_use":
                continue
            inp = part["input"]
            rel = target_rel(inp.get("path", ""))
            if not rel:
                continue
            if part.get("name") == "Write" and "contents" in inp:
                writes[rel] = inp["contents"]
            elif part.get("name") == "StrReplace":
                patches.append((i, rel, inp.get("old_string", ""), inp.get("new_string", "")))
    return writes, patches


def apply_patches(content: str, rel: str, patches) -> str:
    for i, patch_rel, old, new in patches:
        if patch_rel != rel or not old:
            continue
        if old in content:
            content = content.replace(old, new, 1)
        else:
            print(f"  skip patch line {i} on {rel}")
    return content


def main():
    writes, patches = load_transcript_ops()
    files = dict(writes)

    for rel in ("ar_module/pose_sequence.py", "ar_module/pose_validator.py"):
        seed = (BASE / rel).read_text(encoding="utf-8")
        files[rel] = apply_patches(seed, rel, patches)

    for rel, content in files.items():
        dest = BASE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"wrote {rel} ({len(content)} bytes)")

    body = BASE / "ar_module/body_overlay.py"
    if body.exists():
        body.unlink()
        print("deleted ar_module/body_overlay.py")


if __name__ == "__main__":
    main()
