#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./deploy.sh your.vps.ip.address"
  exit 1
fi

VPS_HOST="$1"
DEPLOY_MSG="deploy $(date '+%Y-%m-%d %H:%M:%S %Z')"

git add .
git commit -m "$DEPLOY_MSG" || true
git push

ssh "$VPS_HOST" 'cmd.exe /c "cd /d C:\forex_bot && git pull && taskkill /IM python.exe /F && C:\Python311\python.exe C:\forex_bot\main.py"'
