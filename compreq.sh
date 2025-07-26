#!/bin/env bash

set -xe

branch=compreq

git fetch origin main
git checkout main
git pull --rebase origin main
if ! git checkout -b ${branch}; then
    git branch -D ${branch}
    git checkout -b ${branch}
fi
uv run python -m requirements
uv run task format
if [[ $(git status --porcelain) ]]; then
    uv lock --upgrade
    git \
        -c "user.name=Update requirements bot" \
        -c "user.email=none" \
        commit \
        -am "Update requirements."
    git push origin +${branch}
    gh pr create \
       --title "Update requirements" \
       --body "Automatic update of requirements." \
       --reviewer jesnie \
       || true
fi
