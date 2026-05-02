import subprocess
import json
import os
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
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
# CPU inference is lightning fast for single sentences, so HF Space CPU is perfect here
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# The DB was copied into the same directory as this script by Docker
chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))
moogle_collection = chroma_client.get_collection(name="moogle")
logger.info("Moogle Brain Online.")

mcp = FastMCP(
    "Loogle-Search",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    host="0.0.0.0", 
    port=7860 
)

@mcp.tool()
async def loogle_search(query: str) -> str:
    """
    Searches the local Lean 4 Mathlib library by EXACT TYPE SIGNATURE or EXACT NAME.
    CRITICAL: This tool does NOT understand natural language. Do NOT send queries like "Euler's theorem" or "commutative property".
    
    If you want to find a theorem, you MUST translate your concept into Lean 4 syntax variables using question marks.
    Examples of GOOD queries:
    - "?a + ?b = ?b + ?a" (To find commutativity)
    - "?n * 0 = 0" (To find zero multiplication)
    - "Real.sin ?x" (To find theorems about sine)
    
    Look at your current Lean goal, extract the core type signature, and send that.
    """
    try:
        def run_loogle():
            # Executing: lake exe loogle --json "query"
            return subprocess.run(
                [".lake/build/bin/loogle", "--json", query],
                cwd=LOOGLE_DIR,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=300 # Mathlib index is huge, give it time
            )
            
        process = await asyncio.to_thread(run_loogle)
        
        if process.returncode != 0:
            return f"Loogle Error: {process.stderr}\nTry simplifying your search query."

        result = json.loads(process.stdout)
        
        if "error" in result:
            return f"Loogle Error: {result['error']}"
            
        hits = result.get("hits", [])
        if not hits:
            return "No theorems found matching that query in this Mathlib version."
            
        out = f"Found {len(hits)} results. Top matches:\n\n"
        for hit in hits[:5]: 
            out += f"Name: {hit.get('name')}\n"
            out += f"Type: {hit.get('type')}\n"
            out += f"Module: {hit.get('module')}\n"
            out += "-" * 30 + "\n"
            
        return out
        
    except subprocess.TimeoutExpired:
        return "Error: Loogle search timed out."
    except json.JSONDecodeError:
        return f"Error parsing output. Raw: {process.stdout}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ==========================================
# TOOL 2: MOOGLE (The Concept Engine)
# ==========================================
@mcp.tool()
async def moogle_search(concept: str) -> str:
    """
    Semantic concept search for Lean 4 Mathlib.
    Use this tool when you know the mathematical concept in English (e.g., 'Euler's theorem', 'multiplying by zero', 'topological space') 
    but don't know the exact Lean theorem name or type signature.
    """
    logger.info(f"Moogle Query: '{concept}'")
    try:
        def run_moogle():
            query_vector = embed_model.encode([concept]).tolist()
            return moogle_collection.query(
                query_embeddings=query_vector,
                n_results=5,
                # FIX 1: Explicitly tell Chroma we want the documents and metadata back
                include=["documents", "metadatas"] 
            )
            
        results = await asyncio.to_thread(run_moogle)
        
        # FIX 2: Safely extract the lists so if they are None, Python doesn't crash
        documents = results.get('documents')
        metadatas = results.get('metadatas')
        
        if not documents or not documents[0]:
            return "No semantic matches found. (Warning to agent: The vector database might be empty)."
            
        out = "Semantic Search Results:\n\n"
        for i in range(len(documents[0])):
            # Safely grab the name and doc string
            name = metadatas[0][i].get('name', 'Unknown') if metadatas and metadatas[0] else 'Unknown'
            doc = documents[0][i]
            
            out += f"Theorem Name: {name}\n"
            out += f"Description: {doc}\n"
            out += "-" * 30 + "\n"
            
        return out
        
    except Exception as e:
        logger.error(f"Moogle Error: {str(e)}")
        return f"Moogle Search Error: {str(e)}"

if __name__ == "__main__":
    logger.info("Booting Dual Loogle/Moogle Server on 0.0.0.0:7860...")
    mcp.run(transport="sse")

