#!/usr/bin/env bash
set -euo pipefail
umask 077

config_dir="$HOME/.config/dlm-jailbreak-transfer"
config_file="$config_dir/openrouter.env"
mkdir -p "$config_dir"
chmod 700 "$config_dir"

read -r -s -p "OpenRouter API key (hidden input): " OPENROUTER_API_KEY
printf '\n'
if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "No key entered; nothing written." >&2
    exit 2
fi

tmp_file="$(mktemp "$config_dir/openrouter.env.XXXXXX")"
trap 'rm -f "$tmp_file"' EXIT
printf 'export OPENROUTER_API_KEY=%q\n' "$OPENROUTER_API_KEY" > "$tmp_file"
chmod 600 "$tmp_file"
mv -f "$tmp_file" "$config_file"
unset OPENROUTER_API_KEY
trap - EXIT
echo "Saved the credential file with mode 600: $config_file"
