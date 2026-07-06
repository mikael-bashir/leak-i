import json
import os
import asyncio
import uvicorn
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

    async def boot(self):
        """Spawn loogle. Does NOT wait for the index — _ensure_ready() does."""
        if self.process and self.process.returncode is None:
            return
        logger.info("🚨 [LOOGLE] starting `loogle -i --json` (index load ~3-5 min)…")
        self.is_ready = False
        self.process = await asyncio.create_subprocess_exec(
            "./.lake/build/bin/loogle", "-i", "--json",
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
                return obj
            # Heartbeat / header / other JSON — keep waiting for the real result.
            logger.info(f"[LOOGLE skip] {str(obj)[:80]}")

    async def _ensure_ready(self):
        """Caller MUST hold self.lock. Boot if needed and block until the index
        is loaded (confirmed by a real result to a trivial warm query)."""
        if self.is_ready and self.process and self.process.returncode is None:
            return
        await self.boot()
        logger.info("⏳ [LOOGLE] loading Mathlib index (blocking a warm query)…")
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
        """Kill the (wedged/dead) process so the next call reboots cleanly."""
        self.is_ready = False
        try:
            if self.process:
                self.process.kill()
        except Exception:
            pass

    async def search(self, query: str, timeout: float = QUERY_TIMEOUT) -> dict:
        # Guard trivially-bad input BEFORE it reaches loogle: an empty line makes
        # loogle treat stdin as closed and exit, which would otherwise force a
        # ~2-min cold reboot on the next query.
        if not query or not query.strip():
            return {"error": "Empty query. Give a Lean pattern (e.g. `_ ^ 2`), a name substring in quotes (e.g. \"add_comm\"), or a constant (e.g. `Real.sin`)."}
        async with self.lock:
            try:
                await self._ensure_ready()
            except Exception as e:
                logger.error(f"[LOOGLE] could not start: {e}")
                self._reset()
                return {"error": f"loogle backend failed to start: {e}"}

            await self._drain()
            try:
                self.process.stdin.write((query + "\n").encode("utf-8"))
                await self.process.stdin.drain()
                return await self._read_result(timeout)
            except asyncio.TimeoutError:
                # This ONE query wedged the REPL; reset so it doesn't desync the
                # next query. (Rare now that reads are robust.)
                logger.error(f"[LOOGLE] query timed out ({timeout:.0f}s): {query!r} — resetting")
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

@mcp.tool()
async def loogle_search(query: str) -> str:
    """
    Searches the local Lean 4 Mathlib library for theorems.

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
    Semantic concept search for Lean 4 Mathlib using Natural Language.

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

    # Warm the loogle Mathlib index in the BACKGROUND so the port opens right
    # away (HF marks the Space healthy; moogle is usable immediately). The daemon
    # lock makes the first loogle_search wait behind the warmup instead of racing
    # it — which is what previously kept the index from ever loading.
    asyncio.create_task(loogle_engine.warmup())

    # 1. Grab the standard Starlette ASGI application
    http_app = mcp.sse_app()

    # 2. Add the CORS middleware directly to the app
    http_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*", "mcp-protocol-version", "mcp-session-id"],
        expose_headers=["mcp-session-id"]
    )

    # 3. Start Uvicorn programmatically so it shares the CURRENT event loop
    logger.info("🌐 Serving Dual Loogle/Moogle MCP (SSE) on 0.0.0.0:7860")
    config = uvicorn.Config(
        http_app,
        host="0.0.0.0",
        port=7860,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main_serve())
