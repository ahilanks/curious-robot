"""Offline smoke test for the near-live ckpt transport (mocked hub, no network).

Covers the 2026-06-10 transport changes: learner upload_latest (atomic add+delete-older
commit, hub keeps only the latest ckpt), reclaim_hub_storage (squash + LFS purge), and
the collector's CkptFetcher background thread (slot fill/replace, take(), rejected-set
and champion-floor feedback). Run: python3 tests/test_nearlive_transport.py
"""
import sys
import time
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# ---------------------------------------------------------------- upload_latest
from src import learner_daemon as ld  # noqa: E402

tmp = Path("/tmp/smoke_nearlive_out"); tmp.mkdir(exist_ok=True)
for f in tmp.glob("*"):
    if f.is_file():
        f.unlink()

with mock.patch("huggingface_hub.HfApi") as MockApi:
    api = MockApi.return_value
    api.list_repo_files.return_value = [
        "auto1/ckpt_0000100.pt", "auto1/ckpt_0000200.pt", "auto1/ckpt_0000300.pt",
        "safe15/ckpt_0100000.pt", "buffers/auto1/chunk_1_0001_01000.npz"]
    p = ld.upload_latest({"w": torch.zeros(3)}, tmp, 300, "repo", "auto1",
                         "tok", True, False, prune=True)
    ops = api.create_commit.call_args.kwargs["operations"]
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    adds = [o for o in ops if isinstance(o, CommitOperationAdd)]
    dels = [o for o in ops if isinstance(o, CommitOperationDelete)]
    check("upload: one add of the new ckpt",
          len(adds) == 1 and adds[0].path_in_repo == "auto1/ckpt_0000300.pt")
    check("upload: deletes exactly the two OLDER auto1 ckpts",
          sorted(d.path_in_repo for d in dels)
          == ["auto1/ckpt_0000100.pt", "auto1/ckpt_0000200.pt"])
    check("upload: local file removed after success", p is None
          and not (tmp / "ckpt_0000300.pt").exists())

with mock.patch("huggingface_hub.HfApi") as MockApi:
    api = MockApi.return_value
    api.create_commit.side_effect = RuntimeError("hub down")
    p = ld.upload_latest({"w": torch.zeros(3)}, tmp, 301, "repo", "auto1",
                         "tok", True, False, prune=True)
    check("upload: failure keeps the local fallback",
          p is not None and Path(p).exists())

# ------------------------------------------------------------ reclaim_hub_storage
with mock.patch("huggingface_hub.HfApi") as MockApi:
    api = MockApi.return_value
    api.list_repo_files.return_value = ["auto1/ckpt_0000300.pt", "safe15/ckpt_0100000.pt"]

    def lfs(name):
        o = types.SimpleNamespace(); o.filename = name; return o

    api.list_lfs_files.return_value = [
        lfs("auto1/ckpt_0000100.pt"), lfs("auto1/ckpt_0000300.pt"),
        lfs("buffers/auto1/chunk_1_0001_01000.npz"), lfs("safe15/ckpt_0100000.pt")]
    ld.reclaim_hub_storage("repo", "auto1", "tok", tmp)
    check("reclaim: history squashed", api.super_squash_history.called)
    purged = api.permanently_delete_lfs_files.call_args.kwargs["lfs_files"]
    check("reclaim: purges old ckpt + consumed chunk, spares tip + other runs",
          sorted(f.filename for f in purged)
          == ["auto1/ckpt_0000100.pt", "buffers/auto1/chunk_1_0001_01000.npz"])

# ------------------------------------------------------------------- CkptFetcher
from src import collect_daemon as cd  # noqa: E402

hub = {"ckpts": [(100, "auto1/ckpt_0000100.pt")]}


def fake_hub_ckpts(repo, name, token):
    return list(hub["ckpts"])


def fake_download(repo_id, filename, token, local_dir):
    p = Path(local_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"ckpt")
    return str(p)


fdir = Path("/tmp/smoke_fetcher"); fdir.mkdir(exist_ok=True)
(fdir / "candidates" / "auto1").mkdir(parents=True, exist_ok=True)
leftover = fdir / "candidates" / "auto1" / "ckpt_0000001.pt"
leftover.write_bytes(b"stale")

with mock.patch.object(cd, "hub_ckpts", fake_hub_ckpts), \
     mock.patch("huggingface_hub.hf_hub_download", fake_download):
    f = cd.CkptFetcher("repo", "auto1", "tok", fdir, 0.05, champion_step=-1,
                       rejected=set())
    check("fetcher: crash leftovers cleaned at boot", not leftover.exists())
    f.start()
    time.sleep(0.4)
    with f.lock:
        slot = f.slot
    check("fetcher: downloads the first candidate into the slot",
          slot is not None and slot[0] == 100 and slot[1].exists())
    old_path = slot[1]
    hub["ckpts"] = [(200, "auto1/ckpt_0000200.pt")]   # learner pruned 100, pushed 200
    time.sleep(0.4)
    with f.lock:
        slot = f.slot
    check("fetcher: fresher ckpt replaces an unconsumed slot, stale file unlinked",
          slot is not None and slot[0] == 200 and not old_path.exists())
    cand = f.take()
    check("fetcher: take() pops the candidate and empties the slot",
          cand is not None and cand[0] == 200 and f.take() is None)
    check("fetcher: handed-out step raises the poll floor (no probation re-download)",
          f.champion_step == 200)
    f.note(rejected_step=200)
    time.sleep(0.4)
    check("fetcher: rejected step is never re-downloaded", f.take() is None)
    hub["ckpts"] = [(150, "auto1/ckpt_0000150.pt"), (200, "auto1/ckpt_0000200.pt")]
    f.note(champion_step=180)
    time.sleep(0.4)
    check("fetcher: nothing newer than champion -> empty slot", f.take() is None)

print()
bad = [n for n, c in ok if not c]
print(f"{len(ok) - len(bad)}/{len(ok)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
