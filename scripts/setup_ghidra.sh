#!/usr/bin/env bash
# Downloads the latest Ghidra release, unpacks it to ./ghidra/, and writes GHIDRA_HOME to .env.
# Requires: curl, unzip, java 21+

set -euo pipefail

GHIDRA_DIR="$(pwd)/ghidra"
ENV_FILE="$(pwd)/.env"

# -- Java check --
if ! command -v java &>/dev/null; then
    echo "ERROR: java not found. Install Java 21 (e.g. sudo dnf install java-21-openjdk)." >&2
    exit 1
fi

JAVA_VER=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d. -f1)
if [[ "$JAVA_VER" -lt 21 ]]; then
    echo "ERROR: Java 21+ required, found Java $JAVA_VER." >&2
    exit 1
fi
echo "Java $JAVA_VER OK"

# -- Fetch latest release tag from GitHub --
LATEST=$(curl -fsSL "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/latest" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": "\(.*\)".*/\1/')

if [[ -z "$LATEST" ]]; then
    echo "ERROR: Could not determine latest Ghidra release tag. Check network/GitHub API rate limit." >&2
    exit 1
fi
echo "Latest Ghidra release: $LATEST"

# Tag format: Ghidra_11.x.x_build_DATE → asset name: ghidra_11.x.x_PUBLIC_DATE.zip
VERSION=$(echo "$LATEST" | sed 's/Ghidra_//' | sed 's/_build.*//')
ZIP_NAME="ghidra_${VERSION}_PUBLIC.zip"

DOWNLOAD_URL=$(curl -fsSL "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/latest" \
    | grep '"browser_download_url"' \
    | grep '\.zip"' \
    | head -1 \
    | sed 's/.*"browser_download_url": "\(.*\)".*/\1/')

if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "ERROR: Could not find ZIP download URL in release assets." >&2
    exit 1
fi
echo "Download URL: $DOWNLOAD_URL"

# -- Download --
mkdir -p "$GHIDRA_DIR"
ZIP_PATH="$GHIDRA_DIR/$ZIP_NAME"
if [[ -f "$ZIP_PATH" ]]; then
    echo "Archive already downloaded: $ZIP_PATH"
else
    echo "Downloading $ZIP_NAME ..."
    curl -fSL -o "$ZIP_PATH" "$DOWNLOAD_URL"
fi

# -- Unpack --
UNPACKED=$(find "$GHIDRA_DIR" -maxdepth 1 -type d -name "ghidra_*_PUBLIC" | head -1)
if [[ -n "$UNPACKED" ]]; then
    echo "Already unpacked: $UNPACKED"
else
    echo "Unpacking ..."
    unzip -q "$ZIP_PATH" -d "$GHIDRA_DIR"
    UNPACKED=$(find "$GHIDRA_DIR" -maxdepth 1 -type d -name "ghidra_*_PUBLIC" | head -1)
fi
echo "Ghidra home: $UNPACKED"

# Verify sleigh compiler exists
SLEIGH="$UNPACKED/support/sleigh"
if [[ ! -f "$SLEIGH" ]]; then
    echo "WARNING: sleigh compiler not found at $SLEIGH — check Ghidra version." >&2
fi

# -- Write .env --
if [[ -f "$ENV_FILE" ]]; then
    # Update existing GHIDRA_HOME line or append
    if grep -q "^GHIDRA_HOME=" "$ENV_FILE"; then
        sed -i "s|^GHIDRA_HOME=.*|GHIDRA_HOME=$UNPACKED|" "$ENV_FILE"
    else
        echo "GHIDRA_HOME=$UNPACKED" >> "$ENV_FILE"
    fi
else
    cat > "$ENV_FILE" <<EOF
# Ollama (same keys as docquery)
EMBED_PROVIDER=ollama
EMBED_MODEL=embeddinggemma:latest
EMBED_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
LLM_BASE_URL=http://localhost:11434

# Ghidra
GHIDRA_HOME=$UNPACKED
EOF
fi

echo ""
echo "Setup complete. GHIDRA_HOME=$UNPACKED"
echo "Run: source .env  (or set GHIDRA_HOME in your shell)"
