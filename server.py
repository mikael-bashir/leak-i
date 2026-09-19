import json
import os
import re
import asyncio
import uvicorn
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.cors import CORSMiddleware
import nest_asyncio
import chromadb
from sentence_transformers import SentenceTransformer
import logging

nest_asyncio.apply()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Point exactly to where Docker built it
HOME = os.environ.get("HOME", "/home/user")
LOOGLE_DIR = os.path.join(HOME, "loogle")
# The tree loogle is built against and indexes (a path dependency of loogle).
TENGOKU_DIR = os.environ.get("TENGOKU_DIR", os.path.join(HOME, "tengoku"))

# --- INITIALIZE MOOGLE BRAIN ---
logger.info("Loading Embedding Model and ChromaDB for Moogle...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))
moogle_collection = chroma_client.get_collection(name="moogle")
logger.info("Moogle Brain Online.")

# Create your FastMCP server
mcp = FastMCP(
    "Leak-I",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)

# Per-query cap (warm index answers in <5s; broad queries can be slow). This is a
# backstop, not the normal path — and a timeout NO LONGER kills the daemon on its
# own unless we truly can't resync, so one slow query can't poison the next.
QUERY_TIMEOUT = 45.0
# The first query has to load all of Mathlib's index into RAM — minutes on a
# small CPU. The warm query blocks up to this long.
INDEX_LOAD_TIMEOUT = 900.0


# ==========================================
# LOOGLE DAEMON (Persistent Background Process)
# ==========================================
# WHY THIS REWRITE: the old daemon read a SINGLE stdout line per query and
# json.loads'd it. But loogle interleaves non-result lines on stdout (a banner,
# and heartbeat JSON while a query runs), so readline grabbed the wrong line →
# "Error parsing Loogle output". Worse, any 15s timeout / IO hiccup called
# process.kill(), so every subsequent call cold-rebooted loogle (~3-5 min index
# load) — a permanent reboot→parse-fail loop where even the tool's own example
# queries failed. Now we DRAIN stale output, send the query, then read until a
# real loogle result object ({"hits": …} / {"error": …}) arrives, skipping
# banners and heartbeats and tolerating multi-line JSON.
class LoogleDaemon:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.is_ready = False
        # loogle emits exactly one result object per query line. When a query is
        # abandoned (times out) we DON'T kill the still-warm daemon — instead we
        # count the abandoned query here so its late-arriving result is discarded
        # (not mis-returned) before the next query's real result is read.
        self._pending_abandoned = 0

    async def boot(self):
        """Spawn loogle. Does NOT wait for the index — _ensure_ready() does."""
        if self.process and self.process.returncode is None:
            return
        logger.info("🚨 [LOOGLE] starting `loogle -i --json` (index load ~3-5 min)…")
        self.is_ready = False
        self.process = await asyncio.create_subprocess_exec(
            # Index the whole Tengoku tree: the seeded root plus every library
            # of verified additions (Tengoku/All.lean is the tools' entry point).
            "./.lake/build/bin/loogle", "-i", "--json", "--module", os.environ.get("TENGOKU_MODULE", "Tengoku.All"),
            cwd=LOOGLE_DIR,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(self._log_stderr(self.process))

    async def _log_stderr(self, proc):
        if not proc or not proc.stderr:
            return
        while True:
            try:
                line = await proc.stderr.readline()
                if not line:
                    break
                logger.info(f"[LOOGLE-STDERR] {line.decode('utf-8', 'replace').rstrip()}")
            except Exception:
                break

    async def _drain(self):
        """Consume any pending/stale stdout (leftover result, heartbeats) so the
        next read is for the query we are about to send."""
        if not self.process or not self.process.stdout:
            return
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=0.15)
                if not line:
                    break
            except asyncio.TimeoutError:
                break

    async def _read_result(self, timeout: float) -> dict:
        """Read stdout until loogle emits a real result object (has 'hits' or
        'error'), skipping banners/prompts/heartbeats and tolerating multi-line
        JSON. Raises asyncio.TimeoutError or EOFError."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        pending = ""
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=remaining)
            if not line:
                raise EOFError("loogle closed stdout")
            s = line.decode("utf-8", "replace").strip()
            if not s:
                continue
            candidate = (pending + s) if pending else s
            try:
                obj = json.loads(candidate)
                pending = ""
            except json.JSONDecodeError:
                # Either a multi-line JSON still arriving, or a non-JSON banner.
                if candidate.lstrip()[:1] in ("{", "["):
                    pending = candidate  # accumulate the rest of the object
                else:
                    logger.info(f"[LOOGLE noise] {candidate[:120]}")
                    pending = ""
                continue
            if isinstance(obj, dict) and ("hits" in obj or "error" in obj):
                # A real result object. If earlier queries were abandoned on
                # timeout, their results surface here first — discard exactly that
                # many so THIS query gets its own answer, not a stale one.
                if self._pending_abandoned > 0:
                    self._pending_abandoned -= 1
                    logger.info(
                        f"[LOOGLE] discarded late result from an abandoned query "
                        f"({self._pending_abandoned} still pending)"
                    )
                    continue
                return obj
            # Heartbeat / header / other JSON — keep waiting for the real result.
            logger.info(f"[LOOGLE skip] {str(obj)[:80]}")

    async def _ensure_ready(self):
        """Caller MUST hold self.lock. Boot if needed and block until the index
        is loaded (confirmed by a real result to a trivial warm query)."""
        if self.is_ready and self.process and self.process.returncode is None:
            return
        await self.boot()
        logger.info("⏳ [LOOGLE] loading the Tengoku index (blocking a warm query)…")
        await self._drain()
        self.process.stdin.write(b"Nat.add_comm\n")
        await self.process.stdin.drain()
        await self._read_result(INDEX_LOAD_TIMEOUT)
        self.is_ready = True
        logger.info("✅ [LOOGLE] index resident — searches are fast now.")

    async def warmup(self):
        """Background pre-load so uvicorn can open the port immediately."""
        async with self.lock:
            try:
                await self._ensure_ready()
            except Exception as e:
                logger.error(f"⚠️ [LOOGLE] warmup failed: {e}")

    def _reset(self):
        """Kill the (wedged/dead) process so the next call reboots cleanly. Only
        for genuinely-dead states (EOF, failed boot, wedged past the safety
        valve) — NEVER for an ordinary slow query, which just drops the warm
        index and forces a multi-minute reload that poisons every later query."""
        self.is_ready = False
        # A fresh process has no in-flight results, so clear the abandoned count.
        self._pending_abandoned = 0
        try:
            if self.process:
                self.process.kill()
        except Exception:
            pass

    @staticmethod
    def _reject_reason(query: str) -> str | None:
        """Reject queries that are pathological for loogle BEFORE they reach it —
        one bad query used to stall the daemon for everyone. A bare integer
        (e.g. "1680") makes loogle elaborate it as a term and burn its whole
        heartbeat budget; that exact query caused the outage we're fixing."""
        q = query.strip()
        if re.fullmatch(r"[+-]?\d+", q):
            return (
                "Bare numbers aren't searchable — loogle needs a TYPE PATTERN, a "
                'name substring in quotes, or a constant. Try e.g. `Nat.factorial`, '
                '`"add_comm"`, or a pattern like `_ ^ 2 + _`.'
            )
        return None

    async def search(self, query: str, timeout: float = QUERY_TIMEOUT) -> dict:
        # Guard trivially-bad input BEFORE it reaches loogle: an empty line makes
        # loogle treat stdin as closed and exit, which would otherwise force a
        # ~2-min cold reboot on the next query.
        if not query or not query.strip():
            return {"error": "Empty query. Give a Lean pattern (e.g. `_ ^ 2`), a name substring in quotes (e.g. \"add_comm\"), or a constant (e.g. `Real.sin`)."}
        reject = self._reject_reason(query)
        if reject:
            logger.info(f"[LOOGLE] rejected pathological query {query!r} without hitting loogle")
            return {"error": reject}
        async with self.lock:
            try:
                await self._ensure_ready()
            except Exception as e:
                logger.error(f"[LOOGLE] could not start: {e}")
                self._reset()
                return {"error": f"loogle backend failed to start: {e}"}

            # Only drain stale output when nothing is outstanding. If earlier
            # queries were abandoned, their results are accounted for by
            # _pending_abandoned and skipped inside _read_result — draining here
            # would silently eat them and desync the skip count.
            if self._pending_abandoned == 0:
                await self._drain()
            try:
                self.process.stdin.write((query + "\n").encode("utf-8"))
                await self.process.stdin.drain()
                return await self._read_result(timeout)
            except asyncio.TimeoutError:
                # The REPL is SLOW, not dead. Do NOT kill it — killing drops the
                # warm Mathlib index and forces a ~3-5 min cold reload that
                # poisons EVERY following query (the exact outage this fixes).
                # Leave the daemon warm and mark the query abandoned so its late
                # result is discarded before the next one. Only if several pile
                # up (truly wedged) do we accept a one-time reboot.
                self._pending_abandoned += 1
                logger.error(
                    f"[LOOGLE] query slow (> {timeout:.0f}s): {query!r} — abandoned, "
                    f"index kept warm ({self._pending_abandoned} pending)"
                )
                if self._pending_abandoned >= 3:
                    logger.error("[LOOGLE] too many stuck queries — daemon looks wedged, rebooting once")
                    self._reset()
                return {"error": "Query timed out (too broad/complex). Anchor it with a specific constant (e.g. `Nat`, `Real.sin`) or make it narrower."}
            except EOFError:
                logger.error("[LOOGLE] daemon EOF — will reboot next call")
                self._reset()
                return {"error": "loogle backend restarted — retry the query."}
            except Exception as e:
                logger.error(f"[LOOGLE] I/O error: {e}")
                self._reset()
                return {"error": f"loogle I/O error: {e}"}


# Instantiate the daemon globally
loogle_engine = LoogleDaemon()

async def _run(cmd: list[str], cwd: str, timeout: float) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, f"timed out after {timeout:.0f}s: {' '.join(cmd)}"
    return proc.returncode, out.decode("utf-8", errors="replace")


# --- Tree refresh -------------------------------------------------------------
# One implementation behind three doors: the `tengoku_sync` MCP tool, the
# POST /refresh endpoint the nightly cache workflow calls, and the check at
# start-up. `scripts/pin.sh` (in the tree) pins the tree to its newest
# published cache; loogle is then rebuilt against it and restarted so its
# index covers every verified addition.
_refresh = {"running": False, "last_post": 0.0, "last": "", "queued": False, "count": 0, "kept": 0, "swaps": 0, "cold": 0}
# The tree publishes a small "top-up" with every merge (TENGOKU_TOPUPS=1 makes scripts/pin.sh follow
# them), so refresh requests can arrive every few minutes — and loogle needs 3-5 minutes to index the
# tree. So a refresh never takes the running index away: the new loogle is started BESIDE the old one,
# which keeps answering (its Lean process has the old library files mapped; a refresh only renames new
# files over them), and the two are swapped once the new index answers. Requests are never dropped:
# those arriving during a refresh or inside the minimum gap are folded into one deferred refresh.
REFRESH_MIN_GAP = float(os.environ.get("TENGOKU_REFRESH_MIN_GAP", "600"))
BLUE_GREEN = os.environ.get("TENGOKU_BLUE_GREEN", "1") != "0"


def _memory() -> dict:
    """Bytes: the container's limit, its non-reclaimable use, and the running loogle's heap."""
    out = {"limit": None, "anon": None, "loogle": None}
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        out["limit"] = None if raw == "max" else int(raw)
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("anon "):
                out["anon"] = int(line.split()[1])
    except Exception:
        pass
    try:
        proc = loogle_engine.process
        if proc and proc.returncode is None:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("RssAnon:"):
                    out["loogle"] = int(line.split()[1]) * 1024
    except Exception:
        pass
    return out


def _room_for_a_second_index() -> bool:
    m = _memory()
    if not BLUE_GREEN:
        return False
    if None in (m["limit"], m["anon"], m["loogle"]):
        return True  # cannot measure: the Space has 16 GB and one index is a few
    return m["limit"] - m["anon"] > 1.25 * m["loogle"]
# TENGOKU_AUTO_REFRESH=0 turns every door off: for an instance that runs on a
# developer's working tree (which must never be checked out or overwritten).
AUTO_REFRESH = os.environ.get("TENGOKU_AUTO_REFRESH", "1") != "0"
PIN_SH = os.path.join(TENGOKU_DIR, "scripts", "pin.sh")


async def _ensure_pin() -> None:
    """A tree pinned to a cache commit that predates scripts/pin.sh has no copy
    of it: take the newest helper scripts from origin/main first."""
    if os.path.exists(PIN_SH):
        return
    await _run(["git", "fetch", "-q", "origin", "main"], TENGOKU_DIR, 300)
    await _run(["git", "checkout", "-q", "origin/main", "--", "scripts/pin.sh", "scripts/cache.sh"], TENGOKU_DIR, 60)


async def _tree_check() -> tuple[str, str]:
    """('current' | 'newer' | 'unknown', sha-or-detail) — changes nothing."""
    await _ensure_pin()
    rc, out = await _run([PIN_SH, "--check"], TENGOKU_DIR, 300)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    parts = last.split()
    if rc in (0, 3) and len(parts) == 2 and parts[0] in ("current", "newer"):
        return parts[0], parts[1]
    return "unknown", last[:200]


async def _tengoku_sync() -> str:
    if _refresh["running"]:
        return "⏳ tengoku_sync: a refresh is already running"
    global loogle_engine
    _refresh["running"] = True
    try:
        # No lock here: searches keep flowing to the running loogle while the tree moves under it.
        before = (await _run(["git", "rev-parse", "--short", "HEAD"], TENGOKU_DIR, 30))[1].strip()
        await _ensure_pin()
        rc, out = await _run([PIN_SH], TENGOKU_DIR, 3600)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        if rc == 4:
            # pin.sh could not replay the newest state and put back the one we were serving.
            _refresh["kept"] += 1
            _refresh["last"] = f"kept {before}: {tail}"
            return f"↩️ tengoku_sync: {tail} — still serving {before}, index untouched."
        if rc != 0:
            _refresh["last"] = f"failed: {tail}"
            return "❌ tengoku_sync: could not pin the tree to the newest cache\n" + out[-2000:]
        rc, out = await _run(["lake", "build"], LOOGLE_DIR, 3600)
        if rc != 0:
            _refresh["last"] = "failed: loogle does not build against the refreshed tree"
            return "❌ tengoku_sync: loogle does not build against the refreshed tree\n" + out[-2000:]
        after = (await _run(["git", "rev-parse", "--short", "HEAD"], TENGOKU_DIR, 30))[1].strip()
        old = loogle_engine
        warm = old.is_ready and old.process is not None and old.process.returncode is None
        if warm and _room_for_a_second_index():
            fresh = LoogleDaemon()
            await fresh.warmup()  # minutes; the old index answers every search meanwhile
            if not fresh.is_ready:
                if fresh.process and fresh.process.returncode is None:
                    fresh.process.kill()
                _refresh["last"] = f"failed: the index for {after} did not load; still serving the one for {before}"
                return "❌ tengoku_sync: the new index did not load — the old one keeps serving\n"
            loogle_engine = fresh  # new searches go to the new index from here
            async with old.lock:   # a search already running on the old one finishes first
                if old.process and old.process.returncode is None:
                    old.process.kill()
                    await old.process.wait()
                old.process, old.is_ready = None, False
            _refresh["swaps"] += 1
            how = "the new index was loaded beside the old one and swapped in — no search waited"
        else:
            async with old.lock:
                if old.process and old.process.returncode is None:
                    old.process.kill()
                    await old.process.wait()
                old.process, old.is_ready = None, False
                await old.boot()
            asyncio.create_task(old.warmup())
            _refresh["cold"] += 1
            how = "loogle restarted and re-indexing (ready in a few minutes)"
        _refresh["count"] += 1
        _refresh["last"] = f"{before} → {after}"
        return (f"✅ tengoku_sync: tree {before} → {after} ({tail}); {how}. "
                "Moogle's semantic index is a separate embedding job and is not refreshed here.")
    finally:
        _refresh["running"] = False


@mcp.tool()
async def tengoku_sync() -> str:
    """
    Re-index against the newest published Tengoku build cache: pin the tree to
    that cache's commit, unpack it, rebuild loogle against it and restart
    loogle so its index covers every verified addition. Takes minutes (the
    index is rebuilt from the environment). Moogle's semantic index is a
    separate embedding job and is NOT refreshed here.
    """
    if not AUTO_REFRESH:
        return "⛔ tengoku_sync is disabled on this instance (TENGOKU_AUTO_REFRESH=0: it runs on a working tree)."
    return await _tengoku_sync()


async def _refresh_later(delay: float) -> None:
    """The one deferred refresh that stands in for every request folded into it."""
    await asyncio.sleep(delay)
    _refresh["queued"] = False
    if _refresh["running"]:
        _queue_refresh(60)
        return
    try:
        _refresh["last_post"] = time.time()
        status, _ = await _tree_check()
        if status == "newer":
            logger.info((await _tengoku_sync()).splitlines()[0])
    except Exception as e:
        logger.warning(f"deferred refresh failed: {e}")


def _queue_refresh(delay: float) -> bool:
    if _refresh["queued"]:
        return False
    _refresh["queued"] = True
    asyncio.create_task(_refresh_later(max(5.0, delay)))
    return True


async def _refresh_endpoint(request):
    """GET: is a newer cache published than the one loaded? POST: if so, refresh
    in the background. Public on purpose: it can only ever move the tree to a
    cache competemath/tengoku has PUBLISHED, so the most a stranger can do is
    make this server look at GitHub once every five minutes."""
    from starlette.responses import JSONResponse
    head = (await _run(["git", "rev-parse", "HEAD"], TENGOKU_DIR, 30))[1].strip()
    if request.method == "GET":
        status, sha = await _tree_check()
        return JSONResponse({"status": status, "pinned": head, "newest": sha, "refreshing": _refresh["running"], "queued": _refresh["queued"],
                             "last": _refresh["last"], "refreshes": _refresh["count"], "kept": _refresh["kept"], "swaps": _refresh["swaps"], "cold": _refresh["cold"],
                             "index_ready": loogle_engine.is_ready, "memory": _memory(), "topups": os.environ.get("TENGOKU_TOPUPS", "0")})
    if not AUTO_REFRESH:
        return JSONResponse({"status": "disabled", "pinned": head}, status_code=403)
    now = time.time()
    if _refresh["running"]:
        _queue_refresh(60)
        return JSONResponse({"status": "queued", "why": "a refresh is running", "pinned": head}, status_code=202)
    if now - _refresh["last_post"] < REFRESH_MIN_GAP:
        _queue_refresh(REFRESH_MIN_GAP - (now - _refresh["last_post"]))
        return JSONResponse({"status": "queued", "why": "inside the minimum gap", "pinned": head}, status_code=202)
    _refresh["last_post"] = now
    status, sha = await _tree_check()
    if status != "newer":
        return JSONResponse({"status": status, "pinned": head, "newest": sha})
    asyncio.create_task(_tengoku_sync())
    return JSONResponse({"status": "refreshing", "pinned": head, "newest": sha}, status_code=202)


async def _startup():
    """At start: if a newer cache was published since this image was built (a
    nightly went by while the Space slept), move onto it before indexing."""
    if not AUTO_REFRESH:
        logger.info("🌳 Tree auto-refresh is off (TENGOKU_AUTO_REFRESH=0)")
        await loogle_engine.warmup()
        return
    try:
        status, sha = await _tree_check()
    except Exception as e:  # never let the check keep the service from warming up
        logger.warning(f"tree check failed: {e}")
        status, sha = "unknown", str(e)[:120]
    if status == "newer":
        logger.info(f"🌱 A newer Tengoku cache is published ({sha[:12]}) — refreshing before indexing…")
        result = await _tengoku_sync()
        logger.info(result.splitlines()[0])
        if result.startswith("✅"):
            return  # _tengoku_sync restarted loogle and started the warm-up
    else:
        logger.info(f"🌳 Tree check: {status} {sha[:12]}")
    await loogle_engine.warmup()


@mcp.tool()
async def loogle_search(query: str) -> str:
    """
    Searches the Tengoku tree (Mathlib and every other seeded library, plus verified additions) for theorems by name or type pattern.

    CRITICAL LEAN SYNTAX RULES:
    1. Do NOT use natural language.
    2. Use standard quotes for substrings (e.g., "sq", "pi"). Do NOT manually escape them with backslashes.
    3. ALWAYS anchor your searches with specific Lean constants (e.g., `Nat`, `Real.sin`, `0`) or specific metavariables (e.g., `?a`, `?b`) to keep searches computationally feasible and fast.
    AVOID queries like (_ + _ = _ + _).
    4. Use subexpressions when you need a wider net, e.g. _ * (_ ^ _) finds all lemmas whose statement
    includes a product, with one number raised to a power.
    5. Pattern searches with paramaters are order invariant.
    6. You can search by lemma/theorem conclusion (e.g. |- tsum _ = _ * tsum _), lemma name substring (e.g. "differ"), or by Lean4 constants (e.g. "Real.sin")
    7. You can use comma to enforce multiple filters.

    Examples of good queries:
    - Real.sqrt ?a * Real.sqrt ?a
    - Real.sin, "pi"
    - "add_comm"
    - (?a -> ?b) -> List ?a -> List ?b
    - _ ^ 2 - _ ^ 2, |- _ = _ * _, "sq"
    """
    logger.info(f"Agent requested Loogle search: '{query}'")
    result = await loogle_engine.search(query)

    if not isinstance(result, dict):
        return "Loogle Error: unexpected backend response."
    if result.get("error"):
        return f"Loogle Error: {result['error']}"

    hits = result.get("hits", [])
    if not hits:
        return "No theorems found matching that query. Try a broader pattern, a name substring in quotes, or moogle_search to discover the right name."

    count = result.get("count", len(hits))
    out = f"Found {count} results. Top matches:\n\n"
    for hit in hits[:5]:
        out += f"Name: {hit.get('name')}\nType: {hit.get('type')}\n---\n"
    return out

# ==========================================
# TOOL 2: MOOGLE (The Concept Engine)
# ==========================================
@mcp.tool()
async def moogle_search(concept: str) -> str:
    """
    Semantic concept search over Lean 4 declarations using natural language (index built from Mathlib docstrings; the same declaration names exist in Tengoku).

    Use this tool when you know the mathematical concept in English but don't know
    the exact Lean theorem name or type signature.

    GUIDELINES:
    1. Describe concepts in plain English (e.g., 'mean value theorem', 'multiplying by zero').
    2. Do NOT use Lean wildcards (_), metavariables (?a), or code snippets.
    3. Use this tool to DISCOVER naming conventions (e.g., finding that 'square' is
       often called 'sq' or 'mul_self' in Mathlib).
    4. Once you find a theorem name or naming pattern, PIVOT to loogle_search for
       the exact type signature or to find variations.

    Examples:
    - "Difference of squares"
    - "Intermediate Value Theorem"
    - "Triangle inequality for complex numbers"
    """
    logger.info(f"Moogle Query: '{concept}'")
    try:
        def run_moogle():
            query_vector = embed_model.encode([concept]).tolist()
            return moogle_collection.query(
                query_embeddings=query_vector,
                n_results=10,
                include=["documents", "metadatas"]
            )

        results = await asyncio.to_thread(run_moogle)

        documents = results.get('documents')
        metadatas = results.get('metadatas')

        if not documents or not documents[0]:
            return "No semantic matches found. (Warning to agent: The vector database might be empty)."

        out = "Semantic Search Results:\n\n"
        for i in range(len(documents[0])):
            name = metadatas[0][i].get('name', 'Unknown') if metadatas and metadatas[0] else 'Unknown'
            doc = documents[0][i]

            out += f"Theorem Name: {name}\n"
            out += f"Description: {doc}\n"
            out += "-" * 30 + "\n"

        return out

    except Exception as e:
        logger.error(f"Moogle Error: {str(e)}")
        return f"Moogle Search Error: {str(e)}"

async def main_serve():
    logger.info("Booting Leak-I (Loogle + Moogle)…")

    # Warm the loogle Tengoku index in the BACKGROUND so the port opens right
    # away (HF marks the Space healthy; moogle is usable immediately). The daemon
    # lock makes the first loogle_search wait behind the warmup instead of racing
    # it — which is what previously kept the index from ever loading.
    asyncio.create_task(_startup())

    # 1. Grab the standard Starlette ASGI application
    http_app = mcp.sse_app()
    http_app.add_route("/refresh", _refresh_endpoint, methods=["GET", "POST"])

    # 2. Add the CORS middleware directly to the app
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*", "mcp-protocol-version", "mcp-session-id"],
        expose_headers=["mcp-session-id"]
    )

    # 3. Start Uvicorn programmatically so it shares the CURRENT event loop
    port = int(os.environ.get("PORT", "7860"))
    logger.info(f"🌐 Serving Dual Loogle/Moogle MCP (SSE) on 0.0.0.0:{port}")
    config = uvicorn.Config(
        http_app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main_serve())
