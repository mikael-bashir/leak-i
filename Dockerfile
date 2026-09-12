# 1. Base Image (Matching Pantograph)
FROM ubuntu:22.04

# 2. CREATE THE GUEST USER
RUN useradd -m -u 1000 user

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y \
    curl git build-essential python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# 4. Switch to the unprivileged user
USER user
ENV HOME=/home/user
ENV PATH="${HOME}/.local/bin:${HOME}/.elan/bin:${PATH}"

# 5. Install Lean (elan) & Python package manager (uv)
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 6. Clone Loogle
WORKDIR ${HOME}
RUN git clone https://github.com/nomeata/loogle.git
WORKDIR ${HOME}/loogle
RUN git checkout ceaefdb

# 7. VERSION LOCKING (The Magic Trick)
# We overwrite Loogle's default toolchain with the toolchain the rest of the
# Leak fleet is pinned to.
RUN echo 'leanprover/lean4:v4.34.0-rc2' > lean-toolchain

# Update lake to pull the Mathlib version associated with v4.34.0-rc2
# The environment loogle indexes is the Tengoku tree — one self-contained
# library seeded from Mathlib — pulled in as loogle's single dependency in
# place of Mathlib. loogle's own sources import Mathlib modules by their old
# names; those map 1:1 onto the tree (Mathlib.X -> Tengoku.X, Batteries.X ->
# Tengoku.Std.X).
RUN sed -i 's|^require mathlib from git .*$|require tengoku from git "https://github.com/competemath/tengoku" @ "main"|' lakefile.lean && \
    grep -rl "import Mathlib\|import Batteries" Loogle Loogle.lean Tests.lean 2>/dev/null | xargs -r sed -i -E 's/^(import[[:space:]]+)Mathlib\b/\1Tengoku/; s/^(import[[:space:]]+)Batteries\b/\1Tengoku.Std/'
RUN lake update

# The tree's published build cache replaces `lake exe cache get`. gh needs a
# token to read release assets at build time: pass GH_TOKEN as a build secret.
USER root
RUN apt-get update && apt-get install -y zstd gh && rm -rf /var/lib/apt/lists/*
USER user
RUN --mount=type=secret,id=GH_TOKEN,env=GH_TOKEN cd .lake/packages/tengoku && scripts/cache.sh get

# Compile loogle against the tree
RUN lake build

# 8. Setup Python App Environment
WORKDIR ${HOME}/app
COPY --chown=user . ${HOME}/app

RUN uv python install 3.11
RUN uv venv --python 3.11 ${HOME}/app/.venv
ENV PATH="${HOME}/app/.venv/bin:${PATH}"

# CPU-only torch first: sentence-transformers pulls torch transitively, and
# without this it resolves PyPI's default CUDA build (2GB+ of nvidia-*
# wheels) on this CPU-only hardware, which OOMs the container at boot.
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install FastMCP. "mcp<2" pinned: mcp 2.x renamed FastMCP to MCPServer and
# changed its API, breaking server.py's `from mcp.server.fastmcp import
# FastMCP` import.
RUN uv pip install fastmcp "mcp<2" asyncio nest_asyncio chromadb sentence-transformers

# 9. Environment Variables & Boot
EXPOSE 7860
CMD ["python3", "server.py"]