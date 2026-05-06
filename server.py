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


# ==========================================
# LOOGLE DAEMON (Persistent Background Process)
# ==========================================
class LoogleDaemon:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.is_ready = False

    async def boot(self):
        """Starts Loogle as a persistent background process."""
        if self.process and self.process.returncode is None:
            return

        logger.info("🚨 COLD STARTING LOOGLE DAEMON 🚨")
        logger.info("This will take ~3.5 minutes...")
        self.is_ready = False
        
        # We removed -i and added Lean arguments.
        # --json keeps the output machine-readable.
        self.process = await asyncio.create_subprocess_exec(
            "./.lake/build/bin/loogle", "-i", "--json",
            cwd=LOOGLE_DIR,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        asyncio.create_task(self._log_stderr())

    async def _log_stderr(self):
        if not self.process or not self.process.stderr:
            return
            
        while True:
            try:
                line = await self.process.stderr.readline()
                if not line: break
                
                log_text = line.decode().strip()
                logger.info(f"[DAEMON LOG]: {log_text}")
                
            except Exception as e:
                logger.error(f"Daemon stderr read error: {e}")
                break

    async def search(self, query: str, timeout: int = 15) -> str:
        """Sends a query to the running daemon and waits for the JSON response."""
        async with self.lock:
            # Health check: Is the process dead?
            if not self.process or self.process.returncode is not None:
                logger.warning("Daemon was dead. Rebooting...")
                await self.boot()
                # Wait for reboot to finish by sending a dummy query
                await self._execute_query("1 = 1", timeout=500)

            # Health check: Do the pipes exist?
            if not self.process or not self.process.stdin or not self.process.stdout:
                return "Error: Daemon pipes failed."

            return await self._execute_query(query, timeout)

    async def _execute_query(self, query: str, timeout: int = 15) -> str:
        """Handles the low-level I/O for a query."""
        if self.process is None or self.process.stdout is None or self.process.stdin is None:
            return ""
        try:
            # 1. Clear any stale output from the buffer before sending the new query
            while not self.process.stdout.at_eof():
                try:
                    await asyncio.wait_for(self.process.stdout.readline(), timeout=0.1)
                except asyncio.TimeoutError:
                    break

            # 2. Write the query. We append a newline to trigger the read.
            self.process.stdin.write((query + "\n").encode())
            await self.process.stdin.drain()

            # 3. Wait for the response. 
            # We set a 15-second timeout. If it takes longer, the query is too complex.
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
            
            if not line:
                return "Error: Daemon returned empty line. It may have crashed."
            
            return line.decode().strip()
            
        except asyncio.TimeoutError:
            # If it timed out, the query was too complex (like the heartbeat issue).
            # The REPL is now stuck processing it, so we MUST kill the process 
            # to prevent it from corrupting the next query.
            logger.error(f"Query '{query}' timed out. Killing daemon to reset state.")
            self.process.kill()
            await self.process.wait()
            return "Loogle Error: Query timed out (too complex or broad). Try making the search more specific."
        except Exception as e:
            logger.error(f"I/O Error during query: {e}")
            # Assume state is corrupted on any I/O error
            self.process.kill()
            return f"Loogle I/O Error: {e}"

# Instantiate the daemon globally
loogle_engine = LoogleDaemon()

@mcp.tool()
async def loogle_search(query: str) -> str:
    """
    Searches the local Lean 4 Mathlib library for theorems.
    
    CRITICAL LEAN SYNTAX RULES:
    1. Do NOT use natural language.
    2. Use standard quotes for substrings (e.g., "sq", "pi"). Do NOT manually escape them with backslashes.
    3. NEVER use the `_` wildcard for the right side of a conclusion (e.g., `|- _ = _ * _`). This causes combinatorial explosions and server timeouts.
    4. ALWAYS anchor your searches with specific Lean constants (e.g., `Nat`, `Real.sin`, `0`) or specific metavariables (e.g., `?a`, `?b`) to keep searches computationally feasible and fast.
    
    Examples of good queries:
    - Real.sqrt ?a * Real.sqrt ?a
    - Real.sin, "pi"
    - "add_comm"
    - (?a -> ?b) -> List ?a -> List ?b
    - _ ^ 2 - _ ^ 2, |- _ = _ * _, "sq"
    """
    logger.info(f"Agent requested Loogle search: '{query}'")
    
    raw_result = ""
    try:
        # Ask the persistent daemon instead of spawning a new process!
        raw_result = await loogle_engine.search(query)
        
        if raw_result.startswith("Error:"):
            return raw_result

        # Parse the JSON response
        result = json.loads(raw_result)
        
        if "error" in result:
            return f"Loogle Error: {result['error']}"
            
        hits = result.get("hits", [])
        if not hits:
            return "No theorems found matching that query."
            
        out = f"Found {len(hits)} results. Top matches:\n\n"
        for hit in hits[:5]: 
            out += f"Name: {hit.get('name')}\nType: {hit.get('type')}\n---\n"
            
        return out
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON. Raw output: {raw_result}")
        return "Error parsing Loogle output. See server logs."
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

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
    logger.info("Booting Loogle Daemon...")
    await loogle_engine.boot()
    
    logger.info("⏳ Warming up Mathlib index (This will take ~5 minutes)...")
    logger.info("The server will not open port 7860 until this is completely finished.")
    
    # Send a dummy query to force the daemon to fully load the index into RAM.
    # Python will hang right here until Loogle spits out the JSON answer.
    await loogle_engine.search("1 = 1", 500)
    
    logger.info("✅ Mathlib fully loaded into RAM! Fast MCP searches are now available.")

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
    # This prevents the "Task attached to a different loop" crashes.
    logger.info("Booting up Dual Loogle/Moogle environment...")
    config = uvicorn.Config(
        http_app, 
        host="0.0.0.0", 
        port=7860,
        proxy_headers=True,               # Trust X-Forwarded-* headers
        forwarded_allow_ips="*",
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # asyncio.run handles creating the master event loop for both Loogle and Uvicorn
    asyncio.run(main_serve())