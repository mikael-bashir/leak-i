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
RUN git clone https://github.com/leanprover-community/loogle.git
WORKDIR ${HOME}/loogle
RUN git checkout ceaefdb

# 7. VERSION LOCKING (The Magic Trick)
# We overwrite Loogle's default toolchain with your exact Pantograph toolchain.
RUN echo 'leanprover/lean4:v4.29.1' > lean-toolchain

# Update lake to pull the Mathlib version associated with v4.29.1
RUN lake update

# CRITICAL: Fetch pre-compiled Mathlib binaries for v4.29.1 so HF doesn't timeout
RUN lake exe cache get

# Compile the Loogle executable against the locked environment
RUN lake build

# 8. Setup Python App Environment
WORKDIR ${HOME}/app
COPY --chown=user . ${HOME}/app

RUN uv python install 3.11
RUN uv venv --python 3.11 ${HOME}/app/.venv
ENV PATH="${HOME}/app/.venv/bin:${PATH}"

# Install FastMCP
RUN uv pip install fastmcp asyncio nest_asyncio

# 9. Environment Variables & Boot
EXPOSE 7860
CMD ["python3", "server.py"]