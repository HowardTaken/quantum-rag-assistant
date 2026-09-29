"""Persistent conversation-state storage for the agent.

The default LangGraph MemorySaver keeps thread history in a plain Python dict, so it's
gone the moment the process restarts -- including a routine crash/restart that isn't a
full redeploy. A SQLite-backed checkpointer survives that: same durability as any other
local file, zero extra infrastructure to run. It does NOT survive an ephemeral-disk
redeploy (e.g. Render's free tier wipes the filesystem on each deploy) and does NOT share
state across multiple horizontally-scaled instances -- only an external store (Postgres,
Redis) fixes those, which LangGraph supports via langgraph-checkpoint-postgres if this
ever needs to scale past a single instance.
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


def build_sqlite_checkpointer(path: str) -> SqliteSaver:
    # check_same_thread=False: FastAPI's sync route handlers run in a threadpool, so the
    # connection may be used from a different thread than the one that created it.
    # SqliteSaver serializes access internally, so this is safe.
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
