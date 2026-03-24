#!/bin/bash
set -e

# Bundle OpenSBDD for macOS as a self-contained .dmg.
# Source of truth:
# - QupKake source lives in vendor/QupKake
# - xTB source runtime lives in vendor/xtb-env (preferred)
# - bundle-mac/ is a disposable build artifact and must not be used as a dev source tree
# Usage: bash scripts/bundle-mac.sh
#   or:  npm run dist:mac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUNDLE_DIR="$PROJECT_DIR/bundle-mac"
CONDA_ENV="openmm-metal"
QUPKAKE_ENV="qupkake"

echo "=== OpenSBDD Mac Bundle ==="
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Build TypeScript + Webpack
echo "[1/5] Building app..."
cd "$PROJECT_DIR"
npm run build:electron
npm run build

# Step 2: Pack conda environment
echo "[2/5] Packing conda environment ($CONDA_ENV)..."
CONDA_PACK_FILE="$BUNDLE_DIR/python-env.tar.gz"
QUPKAKE_PACK_FILE="$BUNDLE_DIR/qupkake-python-env.tar.gz"
mkdir -p "$BUNDLE_DIR"

if [ ! -f "$CONDA_PACK_FILE" ]; then
    conda-pack -n "$CONDA_ENV" -o "$CONDA_PACK_FILE" --ignore-editable-packages --ignore-missing-files 2>&1 | tail -5
    echo "  Packed to $CONDA_PACK_FILE ($(du -sh "$CONDA_PACK_FILE" | cut -f1))"
else
    echo "  Using cached pack: $CONDA_PACK_FILE ($(du -sh "$CONDA_PACK_FILE" | cut -f1))"
    echo "  (Delete $CONDA_PACK_FILE to repack)"
fi

echo "  Packing dedicated QupKake environment ($QUPKAKE_ENV)..."
if [ ! -f "$QUPKAKE_PACK_FILE" ]; then
    conda-pack -n "$QUPKAKE_ENV" -o "$QUPKAKE_PACK_FILE" --ignore-editable-packages --ignore-missing-files 2>&1 | tail -5
    echo "  Packed to $QUPKAKE_PACK_FILE ($(du -sh "$QUPKAKE_PACK_FILE" | cut -f1))"
else
    echo "  Using cached pack: $QUPKAKE_PACK_FILE ($(du -sh "$QUPKAKE_PACK_FILE" | cut -f1))"
    echo "  (Delete $QUPKAKE_PACK_FILE to repack)"
fi

# Step 3: Prepare extraResources
echo "[3/5] Preparing bundled resources..."
EXTRA_DIR="$BUNDLE_DIR/extra-resources"
rm -rf "$EXTRA_DIR"
mkdir -p "$EXTRA_DIR/scripts"
mkdir -p "$EXTRA_DIR/python"
mkdir -p "$EXTRA_DIR/qupkake-python"
mkdir -p "$EXTRA_DIR/qupkake-xtb"
mkdir -p "$EXTRA_DIR/cordial"
mkdir -p "$EXTRA_DIR/qupkake-fork"

# Copy Python scripts
cp -r "$PROJECT_DIR/deps/staging/scripts/"*.py "$EXTRA_DIR/scripts/" 2>/dev/null || true
cp -r "$PROJECT_DIR/deps/staging/scripts/configs" "$EXTRA_DIR/scripts/configs" 2>/dev/null || true
cp "$PROJECT_DIR/scripts/score_cordial.py" "$EXTRA_DIR/scripts/" 2>/dev/null || true

# Bundle forked QupKake source if available
if [ -d "$PROJECT_DIR/vendor/QupKake/qupkake" ]; then
    echo "  Including forked QupKake source..."
    rm -rf "$EXTRA_DIR/qupkake-fork"
    mkdir -p "$EXTRA_DIR/qupkake-fork"
    cp -R "$PROJECT_DIR/vendor/QupKake/"* "$EXTRA_DIR/qupkake-fork/"
fi

# Extract conda env
echo "  Extracting conda env (this takes a minute)..."
tar xzf "$CONDA_PACK_FILE" -C "$EXTRA_DIR/python"
tar xzf "$QUPKAKE_PACK_FILE" -C "$EXTRA_DIR/qupkake-python"

# Fix conda-pack prefixes
echo "  Fixing conda-pack prefixes..."
cd "$EXTRA_DIR/python"
if [ -f bin/conda-unpack ]; then
    bash bin/conda-unpack 2>/dev/null || true
fi
cd "$EXTRA_DIR/qupkake-python"
if [ -f bin/conda-unpack ]; then
    bash bin/conda-unpack 2>/dev/null || true
fi
cd "$PROJECT_DIR"

# Bundle source-built xTB beside the dedicated QupKake runtime if available.
QUPKAKE_XTB_ROOT=""
for candidate in \
    "${QUPKAKE_XTB_ROOT:-}" \
    "$PROJECT_DIR/vendor/xtb-env" \
    "$PROJECT_DIR/vendor/xtb-6.4.1/install-openblas" \
    "$PROJECT_DIR/vendor/xtb-6.4.1/install" \
    "$HOME/xtb-6.4.1/install-openblas" \
    "$HOME/xtb-6.4.1/install"; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/xtb" ]; then
        QUPKAKE_XTB_ROOT="$candidate"
        break
    fi
done

if [ -n "$QUPKAKE_XTB_ROOT" ]; then
    echo "  Including native xTB from $QUPKAKE_XTB_ROOT..."
    rm -rf "$EXTRA_DIR/qupkake-xtb"
    mkdir -p "$EXTRA_DIR/qupkake-xtb"
    cp -R "$QUPKAKE_XTB_ROOT/bin" "$EXTRA_DIR/qupkake-xtb/"
    cp -R "$QUPKAKE_XTB_ROOT/lib" "$EXTRA_DIR/qupkake-xtb/" 2>/dev/null || true
    cp -R "$QUPKAKE_XTB_ROOT/share" "$EXTRA_DIR/qupkake-xtb/" 2>/dev/null || true
elif [ -x "$EXTRA_DIR/qupkake-python/bin/xtb" ]; then
    echo "  Falling back to xTB from the dedicated QupKake env..."
    rm -rf "$EXTRA_DIR/qupkake-xtb"
    mkdir -p "$EXTRA_DIR/qupkake-xtb/bin"
    cp "$EXTRA_DIR/qupkake-python/bin/xtb" "$EXTRA_DIR/qupkake-xtb/bin/xtb"
else
    echo "  No xTB binary found for QupKake bundling"
    exit 1
fi

# Bundle CORDIAL if available
CORDIAL_SOURCE=""
for candidate in \
    "${CORDIAL_ROOT:-}" \
    "$PROJECT_DIR/CORDIAL" \
    "$HOME/CORDIAL" \
    "$HOME/cordial" \
    "$HOME/Desktop/CORDIAL" \
    "$HOME/Desktop/FragGen/CORDIAL"; do
    if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -d "$candidate/weights" ] && [ -d "$candidate/modules" ]; then
        CORDIAL_SOURCE="$candidate"
        break
    fi
done

if [ -n "$CORDIAL_SOURCE" ]; then
    echo "  Including CORDIAL from $CORDIAL_SOURCE..."
    rm -rf "$EXTRA_DIR/cordial"
    mkdir -p "$EXTRA_DIR/cordial"
    cp -R "$CORDIAL_SOURCE/"* "$EXTRA_DIR/cordial/"
else
    echo "  CORDIAL not found locally; skipping bundled CORDIAL resources"
fi

# Copy Metal plugin dylibs if available
METAL_BUILD="$HOME/openmm-metal-project/openmm-metal/.build-metal"
if [ -d "$METAL_BUILD" ]; then
    echo "  Including Metal plugin dylibs..."
    for lib in \
        "$METAL_BUILD/platforms/metal/libOpenMMMetal.dylib" \
        "$METAL_BUILD/plugins/amoeba/platforms/metal/libOpenMMAmoebaMetal.dylib" \
        "$METAL_BUILD/plugins/drude/platforms/metal/libOpenMMDrudeMetal.dylib" \
        "$METAL_BUILD/plugins/rpmd/platforms/metal/libOpenMMRPMDMetal.dylib"; do
        if [ -f "$lib" ]; then
            cp "$lib" "$EXTRA_DIR/python/lib/plugins/"
            echo "    $(basename "$lib")"
        fi
    done
fi

echo "  Resources size: $(du -sh "$EXTRA_DIR" | cut -f1)"

# Step 4: Build DMG with electron-builder
echo "[4/5] Packaging app..."
cd "$PROJECT_DIR"
# Build the .app bundle only (not DMG — we'll make it ourselves for large apps)
npx electron-builder --mac dir

echo "[4a/5] Creating styled DMG..."
VERSION=$(node -p "require('./package.json').version")
PRODUCT=$(node -p "require('./package.json').build.productName")
DMG_FILE="$PROJECT_DIR/dist/${PRODUCT}-${VERSION}-mac-arm64.dmg"
rm -f "$DMG_FILE"

# Generate DMG background if missing
BG_FILE="$PROJECT_DIR/assets/dmg-background.png"
if [ ! -f "$BG_FILE" ]; then
  python3 -c "
from PIL import Image, ImageDraw, ImageFont
import math
w, h = 660, 400
img = Image.new('RGBA', (w, h), (35, 25, 20, 255))
draw = ImageDraw.Draw(img)
for y in range(h):
    draw.line([(0, y), (w, y)], fill=(255, 255, 255, int(15 * y / h)))
ember = (255, 140, 50, 180)
pts = []
for t in range(50):
    f = t / 49.0
    pts.append((255 + 150 * f, 180 - 30 * math.sin(f * math.pi)))
for i in range(len(pts)-1):
    draw.line([pts[i], pts[i+1]], fill=ember, width=3)
end, prev = pts[-1], pts[-3]
a = math.atan2(end[1]-prev[1], end[0]-prev[0])
draw.polygon([end, (end[0]-14*math.cos(a-0.4), end[1]-14*math.sin(a-0.4)), (end[0]-14*math.cos(a+0.4), end[1]-14*math.sin(a+0.4))], fill=ember)
try:
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 12)
except: font = ImageFont.load_default()
draw.text((w//2, 340), 'Drag Ember to Applications to install', fill=(255,255,255,130), font=font, anchor='mm')
img.save('$BG_FILE')
" 2>/dev/null || echo "  (background generation skipped)"
fi

if command -v create-dmg &>/dev/null && [ -f "$BG_FILE" ]; then
  create-dmg \
    --volname "$PRODUCT" \
    --volicon "$PROJECT_DIR/assets/icon.icns" \
    --background "$BG_FILE" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 120 \
    --icon "${PRODUCT}.app" 180 180 \
    --hide-extension "${PRODUCT}.app" \
    --app-drop-link 480 180 \
    --no-internet-enable \
    --skip-jenkins \
    "$DMG_FILE" \
    "$PROJECT_DIR/dist/mac-arm64/${PRODUCT}.app" || {
      echo "  Styled DMG failed, falling back to plain DMG..."
      DMG_STAGING="/tmp/${PRODUCT}-dmg-staging"
      rm -rf "$DMG_STAGING"
      mkdir -p "$DMG_STAGING"
      cp -R "$PROJECT_DIR/dist/mac-arm64/${PRODUCT}.app" "$DMG_STAGING/"
      ln -s /Applications "$DMG_STAGING/Applications"
      hdiutil create -volname "$PRODUCT" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_FILE"
      rm -rf "$DMG_STAGING"
    }
else
  echo "  create-dmg not found, using plain DMG..."
  DMG_STAGING="/tmp/${PRODUCT}-dmg-staging"
  rm -rf "$DMG_STAGING"
  mkdir -p "$DMG_STAGING"
  cp -R "$PROJECT_DIR/dist/mac-arm64/${PRODUCT}.app" "$DMG_STAGING/"
  ln -s /Applications "$DMG_STAGING/Applications"
  hdiutil create -volname "$PRODUCT" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_FILE"
  rm -rf "$DMG_STAGING"
fi

# Step 5: Report
echo ""
echo "[5/5] Done!"
echo ""
DMG_FILE=$(ls -t dist/OpenSBDD-*.dmg 2>/dev/null | head -1)
if [ -f "$DMG_FILE" ]; then
    echo "DMG: $DMG_FILE ($(du -sh "$DMG_FILE" | cut -f1))"
else
    echo "DMG location: dist/"
    ls -la dist/*.dmg 2>/dev/null || echo "  (check dist/ for output)"
fi
echo ""
echo "To rebuild after code changes: npm run dist:mac"
echo "To repack conda env: rm $CONDA_PACK_FILE && npm run dist:mac"
