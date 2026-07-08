#!/usr/bin/env bash
set -euo pipefail

state_root="${CHECKOUT_TRANSPORT_STATE_ROOT:-${GITHUB_WORKSPACE}/../.checkout-transport}"
home_dir="${CHECKOUT_TRANSPORT_HOME:-${state_root}/home}"
git_config="${CHECKOUT_TRANSPORT_GIT_CONFIG:-${state_root}/gitconfig}"
ssh_dir="${CHECKOUT_TRANSPORT_SSH_DIR:-${state_root}/ssh}"

mkdir -p "$home_dir" "$ssh_dir"
chmod 700 "$home_dir" "$ssh_dir"

{
  echo "HOME=$home_dir"
  echo "GIT_CONFIG_GLOBAL=$git_config"
} >> "$GITHUB_ENV"

export HOME="$home_dir"
export GIT_CONFIG_GLOBAL="$git_config"

if [[ -n "${CHECKOUT_GH_TOKEN:-}" ]]; then
  git config --file "$git_config" \
    url."https://x-access-token:${CHECKOUT_GH_TOKEN}@github.com/".insteadOf \
    "git@github.com:"
  echo "Configured GitHub HTTPS token checkout rewrite."
  exit 0
fi

if [[ -z "${CHECKOUT_SSH_KEY:-}" ]]; then
  echo "CHECKOUT_GH_TOKEN or CHECKOUT_SSH_KEY is required for GitHub SSH clone URLs." >&2
  exit 2
fi

key_file="${ssh_dir}/github_checkout_key"
printf '%s\n' "$CHECKOUT_SSH_KEY" > "$key_file"
chmod 600 "$key_file"

known_hosts_tmp="$(mktemp)"
ssh-keyscan -p 443 -t rsa,ecdsa,ed25519 ssh.github.com > "$known_hosts_tmp" 2>/dev/null
if [[ ! -s "$known_hosts_tmp" ]]; then
  echo "Unable to scan ssh.github.com host keys." >&2
  exit 2
fi

known_hosts="${ssh_dir}/known_hosts"
awk 'NF >= 3 && $1 !~ /^#/ { print "github.com " $2 " " $3 }' \
  "$known_hosts_tmp" > "$known_hosts"
if [[ ! -s "$known_hosts" ]]; then
  echo "Unable to normalize ssh.github.com host keys for github.com." >&2
  exit 2
fi
chmod 600 "$known_hosts"

ssh_config="${ssh_dir}/config"
cat > "$ssh_config" <<EOF
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  HostKeyAlias github.com
  IdentityFile $key_file
  IdentitiesOnly yes
  StrictHostKeyChecking yes
EOF
chmod 600 "$ssh_config"

ssh_wrapper="${ssh_dir}/github_ssh"
cat > "$ssh_wrapper" <<EOF
#!/usr/bin/env bash
exec ssh -F "$ssh_config" -o UserKnownHostsFile="$known_hosts" -o GlobalKnownHostsFile=/dev/null "\$@"
EOF
chmod 700 "$ssh_wrapper"

echo "GIT_SSH_COMMAND=$ssh_wrapper" >> "$GITHUB_ENV"

ssh_output="$("$ssh_wrapper" -o BatchMode=yes -T git@github.com 2>&1)" && ssh_status=$? || ssh_status=$?
echo "$ssh_output"
if [[ "$ssh_output" == *"Host key verification failed"* ]]; then
  echo "GitHub SSH host key verification failed after explicit checkout setup." >&2
  exit 2
fi
if [[ "$ssh_output" == *"Permission denied"* ]]; then
  echo "GitHub SSH authentication failed for the configured checkout key." >&2
  exit 2
fi
if [[ "$ssh_status" -ne 0 && "$ssh_status" -ne 1 ]]; then
  echo "GitHub SSH probe failed with status $ssh_status." >&2
  exit "$ssh_status"
fi
