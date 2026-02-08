# Body for git-filter-repo's commit_callback(commit)
# NOTE: This file is *not* a standalone script; git-filter-repo wraps it in
# `def commit_callback(commit): ...`.

import json

# Lazy-load mapping once per filter-repo run.
if 'MESSAGE_MAP' not in globals():
    MESSAGE_MAP = {}
    with open('.rewrite/commit_messages.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            MESSAGE_MAP[rec['sha']] = rec['message'].encode('utf-8')

    TARGET_NAME = b'DennySORA'

# commit.original_id may be raw bytes (20) or hex bytes (40); handle both.
oid = commit.original_id
if isinstance(oid, (bytes, bytearray)):
    old_sha = oid.decode('ascii', errors='ignore') if len(oid) == 40 else bytes(oid).hex()
else:
    old_sha = str(oid)

new_msg = MESSAGE_MAP.get(old_sha)
if new_msg is not None:
    commit.message = new_msg

# Set names, preserve emails and timestamps.
commit.author_name = TARGET_NAME
commit.committer_name = TARGET_NAME
