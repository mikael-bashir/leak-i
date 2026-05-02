import subprocess
import json
import os
import asyncio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
import nest_asyncio

nest_asyncio.apply()

# Point exactly to where Docker built it
HOME = os.environ.get("HOME", "/home/user")
LOOGLE_DIR = os.path.join(HOME, "loogle")

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
    Searches your local, v4.29.1-locked Lean 4 Mathlib library for lemmas and theorems.
    You can search by name (e.g., 'Real.sin') or by type signature (e.g., '?a + ?b = ?b + ?a').
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

if __name__ == "__main__":
    print("Booting Local Loogle MCP Server on 0.0.0.0:7860...")
    mcp.run(transport="sse")
