#!/bin/sh
# Installs skull (https://github.com/Kroy665/skull) as a standalone `skull`
# command, with no prior setup required - bootstraps uv itself if it's not
# already installed, exactly the way https://astral.sh/uv/install.sh does,
# rather than assuming the user has ever heard of uv.
#
#   curl -LsSf https://kroy.dev/skull/install.sh | sh
#
# macOS/Linux only (see README.md's Requirements table - raw-terminal input
# in skull itself is POSIX-only, which this script does not change).

set -eu

REPO_URL="https://github.com/Kroy665/skull.git"

info() { printf '\033[1minstall.sh:\033[0m %s\n' "$1"; }
err() { printf '\033[1;31minstall.sh:\033[0m %s\n' "$1" >&2; }

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

info "installing skull from $REPO_URL"
uv tool install --force "git+$REPO_URL"

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
