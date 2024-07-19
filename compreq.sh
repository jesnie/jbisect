#!/bin/env bash

branch=compreq

set -xe

git fetch origin main
git checkout main
git pull --rebase origin main
if ! git checkout -b ${branch}; then
    git branch -D ${branch}
    git checkout -b ${branch}
fi
poetry run python -m requirements
if [[ $(git status --porcelain) ]]; then
    poetry update
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
