FROM ubuntu:22.04

RUN useradd -m -u 1000 user
RUN apt-get update && apt-get install -y \
    curl git build-essential python3 python3-pip python3-venv zstd && \
    rm -rf /var/lib/apt/lists/*

USER user
ENV HOME=/home/user
ENV PATH="${HOME}/.local/bin:${HOME}/.elan/bin:${PATH}"
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR ${HOME}

# The environment loogle indexes is the Tengoku tree — one self-contained Lean 4
# library seeded from Mathlib and other open libraries, plus every verified
# addition — pinned to its newest published build cache so nothing compiles.
# Changing TENGOKU_REFRESH (the installer passes the time) re-clones on a
# rebuild instead of reusing a stale cached clone layer.
ARG TENGOKU_REFRESH=0
RUN echo "refresh ${TENGOKU_REFRESH}" >/dev/null && git clone --filter=blob:none https://github.com/competemath/tengoku.git tengoku
RUN --mount=type=secret,id=GH_TOKEN,env=GH_TOKEN,required=false cd tengoku && scripts/pin.sh \
 && rm -rf .lake/build/ir
ENV TENGOKU_DIR=${HOME}/tengoku

# loogle, built against the tree instead of Mathlib. loogle-tengoku.patch swaps
# its one dependency for the tree (a path dependency, ../tengoku) and renames
# the five import lines (Mathlib -> Tengoku, Batteries.X -> Tengoku.Std.X).
RUN git clone https://github.com/nomeata/loogle.git && cd loogle && git checkout -q ceaefdb
COPY --chown=user loogle-tengoku.patch ${HOME}/loogle/loogle-tengoku.patch
WORKDIR ${HOME}/loogle
RUN git apply loogle-tengoku.patch && cp ../tengoku/lean-toolchain lean-toolchain && lake update && lake build

# Python app (the MCP server; torch/sentence-transformers serve moogle)
WORKDIR ${HOME}/app
COPY --chown=user . ${HOME}/app
RUN uv python install 3.11
RUN uv venv --python 3.11 ${HOME}/app/.venv
ENV PATH="${HOME}/app/.venv/bin:${PATH}"
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install fastmcp "mcp<2" asyncio nest_asyncio chromadb sentence-transformers

EXPOSE 7860
CMD ["python3", "server.py"]
