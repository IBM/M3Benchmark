# Dockerfile for the M3 python_tools MCP server.
#
# Build from apis/m3/python_tools/:
#   cd apis/m3/python_tools && docker build -t m3-sel-slot-server .
#
# Run (mount databases + universe configs from the project root):
#   docker run -d --name m3-sel-slot-server \
#       -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
#       -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
#       m3-sel-slot-server
#
# Use with demo.py:
#   python examples/demo.py --mode docker --container-name m3-sel-slot-server
#
# Or exec manually:
#   docker exec -i -e MCP_DOMAIN=superhero m3-sel-slot-server \
#       python -m apis.m3.python_tools.mcp

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create package init files for Python module resolution
RUN mkdir -p apis/m3 \
    && echo '"""APIs module for enterprise-benchmark."""' > apis/__init__.py \
    && echo '"""M3 APIs module."""' > apis/m3/__init__.py

# Copy the python_tools package (build context = apis/m3/python_tools/)
COPY . apis/m3/python_tools/

# Databases and universe configs are mounted at runtime via -v flags
VOLUME /app/db
VOLUME /app/apis/configs

# Keep container alive for `docker exec -i` usage (MCP STDIO protocol)
CMD ["tail", "-f", "/dev/null"]
