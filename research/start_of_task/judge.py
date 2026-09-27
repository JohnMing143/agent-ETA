"""Run the LLM judge over all batches (resumable). Isolated: temp AGENT_ETA_HOME, no tools, no saved session."""
import glob, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "judged"), exist_ok=True)
env = dict(os.environ, AGENT_ETA_HOME=os.path.join(HERE, "eta_home"))
cost = 0.0
for path in sorted(glob.glob(os.path.join(HERE, "batches", "*.txt"))):
    out = os.path.join(HERE, "judged", os.path.basename(path)[:-4] + ".json")
    if os.path.exists(out):
        continue
    for attempt in (1, 2):
        t0 = time.time()
        p = subprocess.run(["claude", "-p", "--model", "haiku", "--tools", "", "--no-session-persistence",
                            "--output-format", "json", "--system-prompt",
                            "You are a precise estimator. Reply with only the requested JSON."],
                           stdin=open(path), capture_output=True, text=True, env=env, timeout=600)
        try:
            d = json.loads(p.stdout)
            cost += d.get("total_cost_usd") or 0
            m = re.search(r"\[.*\]", d.get("result", ""), re.S)
            arr = json.loads(m.group(0))
            json.dump(arr, open(out, "w"), ensure_ascii=False)
            print("%s: %d items in %.0fs" % (os.path.basename(path), len(arr), time.time() - t0), flush=True)
            break
        except Exception as e:
            print("%s: attempt %d failed (%s) %s" % (os.path.basename(path), attempt, e, p.stderr[-200:]), flush=True)
print("done; reported cost %.2f USD" % cost)
