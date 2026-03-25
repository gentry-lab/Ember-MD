#!/bin/bash
# Copyright (c) 2026 Ember Contributors. MIT License.
set -e

# Bundle Ember for macOS as a self-contained .dmg.
# Source of truth:
# - bundle-mac/ is a disposable build artifact and must not be used as a dev source tree
# Usage: bash scripts/bundle-mac.sh
#   or:  npm run dist:mac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUNDLE_DIR="$PROJECT_DIR/bundle-mac"

echo "=== Ember Mac Bundle ==="
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Build TypeScript + Webpack
echo "[1/5] Building app..."
cd "$PROJECT_DIR"
npm run build:electron
npm run build

# Step 2: Resolve Python runtime source
echo "[2/5] Resolving project-local Python runtime..."
mkdir -p "$BUNDLE_DIR"

PYTHON_ENV_SOURCE=""
for candidate in \
    "${OPENMM_METAL_ENV_ROOT:-}" \
    "$PROJECT_DIR/openmm-metal-env"; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/python" ]; then
        PYTHON_ENV_SOURCE="$candidate"
        break
    fi
done

if [ -z "$PYTHON_ENV_SOURCE" ]; then
    echo "  Required Python runtime not found"
    echo "  Add ./openmm-metal-env or set OPENMM_METAL_ENV_ROOT"
    exit 1
fi

echo "  Using Python runtime from $PYTHON_ENV_SOURCE"

# Step 3: Prepare extraResources
echo "[3/5] Preparing bundled resources..."
EXTRA_DIR="$BUNDLE_DIR/extra-resources"
rm -rf "$EXTRA_DIR"
mkdir -p "$EXTRA_DIR/scripts"
mkdir -p "$EXTRA_DIR/python"
mkdir -p "$EXTRA_DIR/xtb"
mkdir -p "$EXTRA_DIR/cordial"

# Copy Python scripts
cp -r "$PROJECT_DIR/deps/staging/scripts/"*.py "$EXTRA_DIR/scripts/" 2>/dev/null || true
cp -r "$PROJECT_DIR/deps/staging/scripts/configs" "$EXTRA_DIR/scripts/configs" 2>/dev/null || true
cp -r "$PROJECT_DIR/deps/staging/scripts/fonts" "$EXTRA_DIR/scripts/fonts" 2>/dev/null || true
cp "$PROJECT_DIR/scripts/score_cordial.py" "$EXTRA_DIR/scripts/" 2>/dev/null || true

echo "  Copying Python runtime (this takes a minute)..."
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$PYTHON_ENV_SOURCE/" "$EXTRA_DIR/python/"
else
    cp -R "$PYTHON_ENV_SOURCE/." "$EXTRA_DIR/python/"
fi

echo "  Pruning bundled Python runtime..."
SITE_PKGS="$EXTRA_DIR/python/lib/python3.12/site-packages"

# Remove bytecode caches (regenerated on first import)
find "$EXTRA_DIR/python" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$EXTRA_DIR/python" -name "*.pyc" -delete 2>/dev/null || true

# Remove test suites embedded in packages
find "$SITE_PKGS" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PKGS" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Remove packages not needed at runtime
for pkg in notebook jupyterlab jupyterlab_server jupyterlab_pygments \
           jupyter_client jupyter_core jupyter_server jupyter_server_terminals \
           jupyter_events jupyter_lsp jupyterlab_widgets \
           ipykernel ipywidgets IPython babel statsmodels sqlalchemy \
           pytraj packmol_memgen debugpy jedi pip pygments widgetsnbextension; do
    rm -rf "$SITE_PKGS/$pkg" 2>/dev/null || true
    rm -rf "$SITE_PKGS/${pkg}-"*.dist-info 2>/dev/null || true
done

PRUNED_SIZE=$(du -sh "$EXTRA_DIR/python" | cut -f1)
echo "  Python runtime after pruning: $PRUNED_SIZE"

echo "  Validating bundled Python runtime..."
KMP_DUPLICATE_LIB_OK=TRUE "$EXTRA_DIR/python/bin/python" - <<'PY'
import importlib
required = [
    'openmm',
    'openmmforcefields',
    'openff',
    'vina',
    'meeko',
    'rdkit',
    'pdbfixer',
    'MDAnalysis',
    'propka',
    'torch',
    'pypdf',
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Bundled Python validation failed:\n" + "\n".join(missing))
PY

XTB_SOURCE=""
for candidate in \
    "${XTB_ROOT:-}" \
    "$PROJECT_DIR/xtb-env"; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/xtb" ]; then
        XTB_SOURCE="$candidate"
        break
    fi
done

if [ -n "$XTB_SOURCE" ]; then
    echo "  Including native xTB from $XTB_SOURCE..."
    rm -rf "$EXTRA_DIR/xtb"
    mkdir -p "$EXTRA_DIR/xtb"
    cp -R "$XTB_SOURCE/bin" "$EXTRA_DIR/xtb/"
    cp -R "$XTB_SOURCE/lib" "$EXTRA_DIR/xtb/" 2>/dev/null || true
    cp -R "$XTB_SOURCE/share" "$EXTRA_DIR/xtb/" 2>/dev/null || true
else
    echo "  No xTB binary found for bundling"
    exit 1
fi

# Bundle CORDIAL if available
CORDIAL_SOURCE=""
for candidate in \
    "${CORDIAL_ROOT:-}" \
    "$PROJECT_DIR/CORDIAL"; do
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

    echo "  Validating bundled CORDIAL runtime..."
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$EXTRA_DIR/cordial" \
      "$EXTRA_DIR/python/bin/python" - <<'PY'
import run_protocols
from modules.architectures.model_initializer import ModelInitializer
print("CORDIAL validation OK")
PY
else
    echo "  CORDIAL not found locally; skipping bundled CORDIAL resources"
fi

# Copy Metal plugin dylibs if available
METAL_BUILD=""
for candidate in \
    "${OPENMM_METAL_BUILD:-}" \
    "$PROJECT_DIR/Ember-Metal/.build-metal" \
    "$PROJECT_DIR/openmm-metal/.build-metal"; do
    if [ -n "$candidate" ] && [ -f "$candidate/platforms/metal/libOpenMMMetal.dylib" ]; then
        METAL_BUILD="$candidate"
        break
    fi
done

if [ -n "$METAL_BUILD" ]; then
    echo "  Including Metal plugin dylibs from $METAL_BUILD..."
    mkdir -p "$EXTRA_DIR/python/lib/plugins"
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
else
    echo "  Required OpenMM Metal plugin build not found"
    echo "  Set OPENMM_METAL_BUILD or place a build at ./Ember-Metal/.build-metal"
    exit 1
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
DMG_FILE=$(ls -t "dist/${PRODUCT}-"*.dmg 2>/dev/null | head -1)
if [ -f "$DMG_FILE" ]; then
    echo "DMG: $DMG_FILE ($(du -sh "$DMG_FILE" | cut -f1))"
else
    echo "DMG location: dist/"
    ls -la dist/*.dmg 2>/dev/null || echo "  (check dist/ for output)"
fi
echo ""
echo "To rebuild after code changes: npm run dist:mac"
echo "Python runtime source: $PYTHON_ENV_SOURCE"
