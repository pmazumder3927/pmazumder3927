"""Build the telemetry block in README.md from cloned repos and cloc/churn JSON.

Runs inside the analyze-code workflow after repos are cloned into ./repos/{public,private}
and after cloc/churn outputs are written to ./output/. Replaces content between
<!-- TELEMETRY START --> and <!-- TELEMETRY END --> markers in README.md.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOS_ROOT = Path("repos")
OUTPUT_DIR = Path("output")
README = Path("README.md")
START = "<!-- TELEMETRY START -->"
END = "<!-- TELEMETRY END -->"

LANG_LABELS = {
    "JavaScript": "js",
    "Java": "java",
    "TypeScript": "ts",
    "JSX": "jsx",
    "Python": "py",
    "PHP": "php",
    "C#": "cs",
    "Dart": "dart",
    "Go": "go",
    "Rust": "rs",
    "C++": "cpp",
    "C": "c",
}

EXCLUDE_PATH = re.compile(
    r"(?:^|/)(?:Library|Temp|obj|Build|build|node_modules|vendor|dist|\.git)(?:/|$)"
    r"|\.(?:json|html|css|svg|md|ps1|scss|csv|prefab|unity|asset|meta|lock|min\.js)$"
    r"|(?:^|/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|poetry\.lock|Gemfile\.lock|go\.sum)$"
)

VERB_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with"}

BOT_NAME_RE = re.compile(r"\[bot\]|github-actions|dependabot|renovate", re.IGNORECASE)
BOT_EMAIL_RE = re.compile(r"users\.noreply\.github\.com$|\[bot\]@", re.IGNORECASE)


def is_bot(name: str, email: str) -> bool:
    return bool(BOT_NAME_RE.search(name) or BOT_EMAIL_RE.search(email))


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_short(n: int) -> str:
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:,}"


def fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    if d >= 1:
        return f"{d}d {h}h"
    if h >= 1:
        return f"{h}h {m}m"
    return f"{m}m"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text() or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def all_repos() -> list[Path]:
    repos: list[Path] = []
    for sub in ("public", "private"):
        d = REPOS_ROOT / sub
        if not d.exists():
            continue
        for r in sorted(d.iterdir()):
            if (r / ".git").exists():
                repos.append(r)
    return repos


def git_log(repo: Path, *args: str, author: str | None = None) -> str:
    cmd = ["git", "-C", str(repo), "log", "--no-merges", *args]
    if author:
        cmd.append(f"--author={author}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.strip().replace(" ", "T", 1))
    except ValueError:
        return None


def collect(repos: list[Path], author: str | None):
    """Walk every repo once, collecting commit timestamps, subjects, and numstat."""
    times: list[datetime] = []
    msgs: list[str] = []
    files_per_commit: list[int] = []
    file_touches: Counter[str] = Counter()  # file path -> # commits

    for repo in repos:
        meta = git_log(repo, "--format=%H%x09%cI%x09%an%x09%ae%x09%s", author=author)
        for line in meta.splitlines():
            parts = line.split("\t", 4)
            if len(parts) < 5:
                continue
            _, ci, an, ae, subject = parts
            if is_bot(an, ae):
                continue
            d = parse_iso(ci)
            if d is None:
                continue
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            times.append(d.astimezone(timezone.utc))
            msgs.append(subject)

        ns = git_log(repo, "--numstat", "--format=__SHA__%H%x09%an%x09%ae", author=author)
        per = 0
        seen = False
        skip = False
        for line in ns.splitlines():
            if line.startswith("__SHA__"):
                if seen and not skip:
                    files_per_commit.append(per)
                per = 0
                seen = True
                header = line[len("__SHA__"):]
                hp = header.split("\t")
                an = hp[1] if len(hp) > 1 else ""
                ae = hp[2] if len(hp) > 2 else ""
                skip = is_bot(an, ae)
                continue
            if skip or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            add, delete, fp = parts[0], parts[1], parts[2]
            if add == "-" or delete == "-":
                continue
            if EXCLUDE_PATH.search(fp):
                continue
            per += 1
            file_touches[fp] += 1
        if seen and not skip:
            files_per_commit.append(per)

    return times, msgs, files_per_commit, file_touches


def peak_hour(times: list[datetime]) -> tuple[int | None, float]:
    if not times:
        return None, 0.0
    hours = Counter(d.hour for d in times)
    h, _ = hours.most_common(1)[0]
    night = sum(1 for d in times if d.hour >= 20 or d.hour < 6)
    return h, night / len(times)


def weekend_split(times: list[datetime]) -> tuple[int, int, int]:
    sat = sum(1 for d in times if d.weekday() == 5)
    sun = sum(1 for d in times if d.weekday() == 6)
    week = sum(1 for d in times if d.weekday() < 5)
    return sat, sun, week


def cadence(times: list[datetime]) -> tuple[float | None, float | None]:
    if len(times) < 2:
        return None, None
    ts = sorted(times)
    gaps = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None, None
    return sum(gaps) / len(gaps), max(gaps)


def streaks(times: list[datetime]) -> tuple[int, int]:
    days = sorted({d.date() for d in times})
    if not days:
        return 0, 0
    longest = run = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    today = datetime.now(timezone.utc).date()
    if (today - days[-1]).days > 1:
        current = 0
    else:
        current = 1
        for i in range(len(days) - 1, 0, -1):
            if (days[i] - days[i - 1]).days == 1:
                current += 1
            else:
                break
    return longest, current


def favorite_verbs(msgs: list[str]) -> list[tuple[str, int]]:
    words: Counter[str] = Counter()
    for m in msgs:
        m = m.strip().lower()
        if not m:
            continue
        first = re.split(r"[\s/.,;:!?()\[\]]+", m, maxsplit=1)[0]
        first = first.strip("\"'`-_")
        if not first or first.isdigit() or len(first) <= 1:
            continue
        if first in VERB_STOPWORDS:
            continue
        words[first] += 1
    return words.most_common(3)


def repo_status(repos: list[Path], author: str | None) -> tuple[int, int, int, int]:
    now = datetime.now(timezone.utc)
    active = dormant = abandoned = 0
    counted = 0
    for repo in repos:
        out = git_log(repo, "--format=%cI%x09%an%x09%ae", author=author)
        dates = []
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3 or is_bot(parts[1], parts[2]):
                continue
            d = parse_iso(parts[0])
            if d is not None:
                dates.append(d)
        if not dates:
            continue
        counted += 1
        last = max(dates)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        delta = now - last
        if delta.days <= 7:
            active += 1
        if delta.days >= 30:
            dormant += 1
        if len(dates) <= 2 and delta.days >= 30:
            abandoned += 1
    return active, counted, dormant, abandoned


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[idx]


def trim_path(p: str, limit: int = 38) -> str:
    if len(p) <= limit:
        return p
    parts = p.split("/")
    if len(parts) <= 1:
        return "…" + p[-(limit - 1):]
    tail = parts[-1]
    out = ".../" + tail
    i = len(parts) - 2
    while i >= 0 and len(out) + len(parts[i]) + 1 <= limit:
        out = parts[i] + "/" + out
        i -= 1
    if len(out) > limit:
        out = "…/" + tail
        if len(out) > limit:
            out = "…" + tail[-(limit - 1):]
    return out


def bar(value: int, ceiling: int, width: int = 20) -> str:
    if ceiling <= 0:
        return "·" * width
    ratio = math.log(1 + value) / math.log(1 + ceiling) if value > 0 else 0
    filled = min(width, int(round(width * ratio)))
    return "█" * filled + "·" * (width - filled)


def lang_total(cloc_pub: dict, cloc_pri: dict, lang: str) -> int:
    def get(d: dict, key: str) -> int:
        return d.get(key, {}).get("code", 0) if isinstance(d.get(key), dict) else 0
    if lang == "Python":
        return (get(cloc_pub, "Python") + get(cloc_pub, "Jupyter Notebook")
                + get(cloc_pri, "Python") + get(cloc_pri, "Jupyter Notebook"))
    return get(cloc_pub, lang) + get(cloc_pri, lang)


def build_block() -> str:
    author = os.getenv("AUTHOR_FILTER") or None

    cloc_pub = load_json(OUTPUT_DIR / "cloc-public.json")
    cloc_pri = load_json(OUTPUT_DIR / "cloc-private.json")
    pub_total = cloc_pub.get("SUM", {}).get("code", 0)
    pri_total = cloc_pri.get("SUM", {}).get("code", 0)
    total = pub_total + pri_total
    pub_pct = round(100 * pub_total / total) if total else 0
    pri_pct = 100 - pub_pct if total else 0

    week_pub = load_json(OUTPUT_DIR / "churn-public-week.json")
    week_pri = load_json(OUTPUT_DIR / "churn-private-week.json")
    week_add = week_pub.get("additions", 0) + week_pri.get("additions", 0)
    week_del = week_pub.get("deletions", 0) + week_pri.get("deletions", 0)
    week_net = week_add - week_del

    all_pub = load_json(OUTPUT_DIR / "churn-public-all.json")
    all_pri = load_json(OUTPUT_DIR / "churn-private-all.json")
    lt_add = all_pub.get("additions", 0) + all_pri.get("additions", 0)
    lt_del = all_pub.get("deletions", 0) + all_pri.get("deletions", 0)

    repos = all_repos()
    times, msgs, fpc, file_touches = collect(repos, author)

    h, night_ratio = peak_hour(times)
    sat, sun, week = weekend_split(times)
    weekend_total = sat + sun + week
    weekend_idx = (sat + sun) / weekend_total if weekend_total else 0
    sat_pct = round(100 * sat / max(sat + sun, 1))
    sun_pct = 100 - sat_pct if (sat + sun) else 0
    avg_gap, max_gap = cadence(times)
    longest_streak, cur_streak = streaks(times)

    fpc_avg = (sum(fpc) / len(fpc)) if fpc else 0
    fpc_p95 = percentile(fpc, 0.95)
    fpc_max = max(fpc) if fpc else 0

    verbs = favorite_verbs(msgs)
    verb_top = f'"{verbs[0][0]}"' if verbs else "—"
    verb_runner = f'"{verbs[1][0]}"' if len(verbs) > 1 else "—"

    active, total_repos, dormant, abandoned = repo_status(repos, author)
    if file_touches:
        path, mt_count = file_touches.most_common(1)[0]
        mt_label = trim_path(path)
    else:
        mt_label, mt_count = "—", 0

    rows: list[tuple[str, int]] = []
    for lang, label in LANG_LABELS.items():
        n = lang_total(cloc_pub, cloc_pri, lang)
        if n > 0:
            rows.append((label, n))
    rows.sort(key=lambda x: -x[1])
    label_w = max((len(l) for l, _ in rows), default=4)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    LBL = 14   # label column width
    COL = 24   # value column width

    def row(label: str, col1: str, col2: str = "") -> str:
        if col2:
            return f"  {label:<{LBL}}{col1:<{COL}}{col2}"
        return f"  {label:<{LBL}}{col1}"

    night_pct = round(night_ratio * 100)
    h_str = f"{h:02d}:00 UTC" if h is not None else "—"
    streak_label = f"{cur_streak} day" + ("" if cur_streak == 1 else "s")

    lines = [
        '<pre><code style="font-family: monospace; font-size: 14px;">',
        f"{today} · telemetry",
        "",
        row("output", f"{fmt_int(total)} loc", f"public {pub_pct}% · private {pri_pct}%"),
        row("past 7d", f"+{fmt_int(week_add)} / -{fmt_int(week_del)}",
            f"net {'+' if week_net >= 0 else ''}{fmt_int(week_net)}"),
        row("lifetime", f"+{fmt_short(lt_add)} / -{fmt_short(lt_del)}"),
        "",
    ]
    for label, n in rows:
        lines.append(f"  {label:<{label_w}}  {bar(n, total)}  {fmt_int(n):>9}")
    lines += [
        "",
        row("peak hour", h_str, f"{night_pct}% past sunset (20:00–06:00)"),
        row("cadence", f"{fmt_dur(avg_gap)} avg gap", f"longest: {fmt_dur(max_gap)}"),
        row("weekend share", f"{round(weekend_idx*100)}%", f"sat {sat_pct}% / sun {sun_pct}%"),
        row("files/commit", f"{fpc_avg:.1f} avg", f"p95 {fpc_p95}, max {fpc_max}"),
        row("commit streak", streak_label, f"longest ever: {longest_streak}"),
        "",
        row("active repos", f"{active} of {total_repos}", f"{dormant} dormant ≥30d"),
        row("abandoned", str(abandoned), "inits without follow-through"),
        row("most-touched", f"{mt_label} ({mt_count}×)" if mt_count else "—"),
        row("favorite verb", verb_top, f"runner up: {verb_runner}"),
        "</code></pre>",
    ]
    return "\n".join(lines)


def main() -> None:
    block = build_block()
    if not README.exists():
        README.write_text(f"{START}\n{block}\n{END}\n")
        return
    text = README.read_text()
    if START in text and END in text:
        text = re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            f"{START}\n{block}\n{END}",
            text,
            flags=re.DOTALL,
        )
    else:
        text = text.rstrip() + f"\n\n{START}\n{block}\n{END}\n"
    README.write_text(text)


if __name__ == "__main__":
    main()
