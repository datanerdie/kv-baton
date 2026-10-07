#!/usr/bin/env bash
# Build Sushi 1.1.1 + kvh/sushi-kvh-import.patch (POST /v1/kvh/import: index an SSD-tier cache entry written
# while the server runs, so a bigdoc handoff needs no Sushi restart). EXPERIMENTAL, local patch, not upstream.
# No Xcode needed: only Sushi's Zig code is compiled; MLX/mlx-c are the prebuilt dylibs + metallib from the
# installed Homebrew sushi 1.1.1, with the mlx-c headers of the commit 1.1.1 pins (56b2d39). Needs brew webp.
#   kvh/build-sushi-kvh.sh [DEST]      (default DEST ~/.local/opt/sushi-1.1.1-kvh)
# Result: DEST/zig-out/bin/sushi (its rpath finds DEST/lib/mlx/lib).
set -euo pipefail
DEST=${1:-$HOME/.local/opt/sushi-1.1.1-kvh}
PATCH=$(cd "$(dirname "$0")" && pwd)/sushi-kvh-import.patch
REL=/opt/homebrew/Cellar/sushi/1.1.1/libexec/lib
[ -f "$REL/libmlxc.dylib" ] || { echo "needs the Homebrew sushi 1.1.1 release (brew install beamivalice/tap/sushi)"; exit 1; }
brew list webp >/dev/null 2>&1 || brew install webp
[ -e "$DEST" ] && { echo "$DEST exists; remove it first"; exit 1; }
git -c advice.detachedHead=false clone -q --branch v1.1.1 --depth 1 https://github.com/beamivalice/sushi.git "$DEST"
cd "$DEST"
git submodule update -q --init --depth 1 lib/mlxc-src
mkdir -p lib/mlx/include/mlx/c lib/mlx/lib
cp lib/mlxc-src/mlx/c/*.h lib/mlx/include/mlx/c/
cp "$REL"/libmlx.dylib "$REL"/libmlxc.dylib "$REL"/libjaccl.dylib "$REL"/mlx.metallib lib/mlx/lib/   # NOT libwebp: link brew's
printf 'mlx=d73eb752ef2e6288fd95b032c0bff0a15a4a9e93 mlxc=56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6 target=26.2\n' > lib/mlx/.version
git -c user.name=kvh -c user.email=kvh@localhost am -q "$PATCH"
./scripts/fetch-zig.sh >/dev/null
.zig-toolchain/zig build -Doptimize=ReleaseFast --prefix "$DEST/zig-out"
"$DEST/zig-out/bin/sushi" --version | grep -v '^\[mem\]'
echo "built $DEST/zig-out/bin/sushi"
