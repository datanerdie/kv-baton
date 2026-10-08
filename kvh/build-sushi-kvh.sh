#!/usr/bin/env bash
# Build Sushi + kvh/sushi-kvh-import.patch (POST /v1/kvh/import: index an SSD-tier cache entry written
# while the server runs, so a bigdoc handoff needs no Sushi restart). EXPERIMENTAL, local patch, not upstream.
# No Xcode needed: only Sushi's Zig code is compiled; MLX/mlx-c are the prebuilt dylibs + metallib of the matching
# release (its mlx-c patch only touches ops.cpp, so the headers of the pinned mlx-c commit fit). Needs brew webp.
#   kvh/build-sushi-kvh.sh [DEST]
#     SUSHI_VERSION  release tag without the v (default 1.2.1); the patch must apply to it
#     SUSHI_REL_LIB  the release's lib folder (default ~/.local/opt/sushi-$SUSHI_VERSION-release/sushi-macos-arm64/lib,
#                    else Homebrew's /opt/homebrew/Cellar/sushi/$SUSHI_VERSION/libexec/lib)
#     DEST           default ~/.local/opt/sushi-$SUSHI_VERSION-kvh
# Result: DEST/zig-out/bin/sushi (its rpath finds DEST/lib/mlx/lib).
# The 1.1.1 build (2026-10-07) used the 1.1.1 version of the patch, in git history before the 1.2.0 port;
# the 1.2.0 build used the patch as it was before the 1.2.1 port (2026-10-08, same code, line offsets only).
set -euo pipefail
VER=${SUSHI_VERSION:-1.2.1}
DEST=${1:-$HOME/.local/opt/sushi-$VER-kvh}
PATCH=$(cd "$(dirname "$0")" && pwd)/sushi-kvh-import.patch
REL=${SUSHI_REL_LIB:-$HOME/.local/opt/sushi-$VER-release/sushi-macos-arm64/lib}
[ -f "$REL/libmlxc.dylib" ] || REL=/opt/homebrew/Cellar/sushi/$VER/libexec/lib
[ -f "$REL/libmlxc.dylib" ] || { echo "needs the sushi $VER release libs (SUSHI_REL_LIB, or brew install beamivalice/tap/sushi)"; exit 1; }
brew list webp >/dev/null 2>&1 || brew install webp
[ -e "$DEST" ] && { echo "$DEST exists; remove it first"; exit 1; }
git -c advice.detachedHead=false clone -q --branch "v$VER" --depth 1 https://github.com/beamivalice/sushi.git "$DEST"
cd "$DEST"
git submodule update -q --init --depth 1 lib/mlxc-src
mkdir -p lib/mlx/include/mlx/c lib/mlx/lib
cp lib/mlxc-src/mlx/c/*.h lib/mlx/include/mlx/c/
cp "$REL"/libmlx.dylib "$REL"/libmlxc.dylib "$REL"/libjaccl.dylib "$REL"/mlx.metallib lib/mlx/lib/   # NOT libwebp: link brew's
pin () { git ls-tree HEAD "$1" | awk '{print $3}'; }
printf 'mlx=%s mlxc=%s target=26.2\n' "$(pin lib/mlx-src)" "$(pin lib/mlxc-src)" > lib/mlx/.version
git -c user.name=kvh -c user.email=kvh@localhost am -q "$PATCH"
./scripts/fetch-zig.sh >/dev/null
.zig-toolchain/zig build -Doptimize=ReleaseFast --prefix "$DEST/zig-out"
"$DEST/zig-out/bin/sushi" --version | grep -v '^\[mem\]'
echo "built $DEST/zig-out/bin/sushi"
