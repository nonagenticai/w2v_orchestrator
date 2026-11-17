#!/usr/bin/env bash
set -euo pipefail

# Validate generated Playwright demo scripts
SCRIPTS_DIR="${1:-output/scripts}"

echo "Validating scripts in $SCRIPTS_DIR..."

# 1. TypeScript compilation check
echo "Step 1: TypeScript compilation..."
cd "$SCRIPTS_DIR"
npx tsc --noEmit 2>&1 || { echo "TypeScript compilation failed"; exit 1; }
echo "  OK"

# 2. Playwright test discovery
echo "Step 2: Playwright test discovery..."
npx playwright test --list --config=playwright.recording.config.ts 2>&1 || { echo "Test discovery failed"; exit 1; }
echo "  OK"

# 3. Headless dry run
echo "Step 3: Headless dry run..."
npx playwright test parts/ --config=playwright.recording.config.ts --reporter=list 2>&1
echo "  Done"

echo "Validation complete."
