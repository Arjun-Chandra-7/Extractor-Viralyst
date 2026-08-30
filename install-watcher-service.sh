#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
install -m 0644 "$root_dir/deploy/viralyst-extractor.service" "$unit_dir/viralyst-extractor.service"
systemctl --user daemon-reload
systemctl --user enable --now viralyst-extractor.service
systemctl --user status viralyst-extractor.service --no-pager
