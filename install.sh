#!/bin/sh
# Installs skull (https://github.com/Kroy665/skull) as a standalone `skull`
# command, with no prior setup required - bootstraps uv itself if it's not
# already installed, exactly the way https://astral.sh/uv/install.sh does,
# rather than assuming the user has ever heard of uv.
#
#   curl -LsSf https://raw.githubusercontent.com/Kroy665/skull/refs/heads/main/install.sh | sh
#
# Installs from the latest tagged GitHub Release, via `uv tool install
# git+...@<tag>` rather than a plain tarball URL - a tarball/built-
# distribution install would silently ignore pyproject.toml's own
# [tool.uv.sources] (uv only reads a package's own source/index config
# when installing from a source tree - a git ref or local path - not from
# a built .tar.gz; see https://github.com/astral-sh/uv/issues/19480),
# which is what pins torch to a CPU-only build on Linux instead of
# pulling several GB of CUDA packages nobody here needs. This does mean
# `git` itself needs to be present on the machine (not a manual clone -
# uv fetches it), which every mainstream OS either ships with or makes a
# one-line install.
#
# macOS/Linux only (see README.md's Requirements table - raw-terminal input
# in skull itself is POSIX-only, which this script does not change).

set -eu

REPO="Kroy665/skull"

info() { printf '\033[1minstall.sh:\033[0m %s\n' "$1"; }
err() { printf '\033[1;31minstall.sh:\033[0m %s\n' "$1" >&2; }

if ! command -v git >/dev/null 2>&1; then
    err "git is required but not found on this machine."
    err "Install it (e.g. 'apt install git', 'brew install git', or via Xcode Command Line Tools on macOS), then re-run this script."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    info "uv not found - installing it first"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # uv's own installer writes to ~/.local/bin (or $CARGO_HOME/bin on some
    # setups) but doesn't export PATH into this already-running shell -
    # needed immediately below to invoke the freshly installed uv itself.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        err "uv installed but not found on PATH - open a new terminal and re-run this script."
        exit 1
    fi
else
    info "uv already installed"
fi

info "looking up the latest release of $REPO"
LATEST_TAG=$(curl -sSf "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -1 | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')

if [ -z "$LATEST_TAG" ]; then
    err "could not determine the latest release tag - check https://github.com/$REPO/releases"
    exit 1
fi

info "installing skull $LATEST_TAG"
uv tool install --force "skull @ git+https://github.com/$REPO@$LATEST_TAG"

BIN_DIR=$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        err "$BIN_DIR is not on your PATH yet."
        err "Add this to your shell profile (~/.zshrc, ~/.bashrc, etc.), then open a new terminal:"
        err "    export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

CONFIG_DIR="$HOME/.config/skull"
if [ "$(uname -s)" = "Darwin" ]; then
    CONFIG_DIR="$HOME/Library/Application Support/skull"
fi

if [ ! -f "$CONFIG_DIR/.env" ]; then
    info "skull installed. One more step before it'll run:"
    info ""
    info "    mkdir -p \"$CONFIG_DIR\""
    info "    cat > \"$CONFIG_DIR/.env\" <<'EOF'"
    info "    QWEN_URL=https://your-qwen-endpoint"
    info "    QWEN_KEY=your-key-here"
    info "    EOF"
    info ""
    info "Then run: skull"
else
    info "skull installed. Config already found at $CONFIG_DIR/.env - run: skull"
fi
