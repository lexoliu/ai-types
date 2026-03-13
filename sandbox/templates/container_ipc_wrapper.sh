MAY_IPC_BIN="$(command -v leash-ipc || true)";
if [ -z "$MAY_IPC_BIN" ]; then echo "leash-ipc command not found in container" >&2; exit 127; fi;
MAY_IPC_DIR="$(mktemp -d)";
cleanup(){ local __may_ipc_status=$?; wait >/dev/null 2>&1 || true; rm -rf "$MAY_IPC_DIR"; return $__may_ipc_status; };
trap cleanup EXIT;
for cmd in {{ escaped_commands }}; do
printf '%s\n' '#!/usr/bin/env bash' "exec \"$MAY_IPC_BIN\" \"$cmd\" \"\$@\"" > "$MAY_IPC_DIR/$cmd";
chmod +x "$MAY_IPC_DIR/$cmd";
done;
export PATH="$MAY_IPC_DIR:$PATH";
hash -r;
export LEASH_IPC_SOCKET="tcp://${MAY_HOST_GATEWAY:-host.docker.internal}:{{ ipc_port }}";
{{ script }}
