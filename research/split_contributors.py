"""Group public session files into one pseudo-user per contributor (by the project path they record)."""
import glob, json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
def who(cwd):
    c = (cwd or "").lower().replace("\\", "/")
    if c.startswith("d:"):
        return "win_d"
    if c.startswith("c:"):
        return "win_c"
    parts = [x for x in c.split("/") if x and x not in ("users", "home", "user")]
    return "_".join(parts[:1] + parts[-1:])[:40] or "unknown"
for p in glob.glob(os.path.join(src, "*.jsonl")):
    cwd = None
    with open(p, errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("cwd"):
                cwd = r["cwd"]
                break
    d = os.path.join(dst, who(cwd), "projects", "x")
    os.makedirs(d, exist_ok=True)
    shutil.copy(p, d)
