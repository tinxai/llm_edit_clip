#!/bin/bash
# usage: launch.sh name "focus text"
P=<repo>; D=$P/panel/$1; mkdir -p $D; cd $D
export AGENT_DIR=$D CUDA_VISIBLE_DEVICES=0 BUDGET=40
PROMPT="$(cat $P/panel/common.md)

## Your AGENT_DIR: $D
## Your focus: $2"
timeout 3000 codex exec -m gpt-6-astra -c model_reasoning_effort=high -c shell_environment_policy.inherit=all --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C $D -o $D/last_message.txt "$PROMPT" > $D/codex.log 2>&1
