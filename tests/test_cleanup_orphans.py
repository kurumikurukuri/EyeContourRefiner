"""tools/data_management/cleanup_orphans.py の削除ガードのテスト。

DB が 0 行を返す・別 DB を読んでしまう、といった状況では「DB に参照が無い画像」が
data/images/ の全画像になり、git 管理外の画像が不可逆に消える。そうした異常な入力では
何も削除せず中断すること、正常な入力ではこれまで通り孤立画像だけを削除することを保証する。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_management import cleanup_orphans as co  # noqa: E402

ATTRIBUTION_HEADER = ["image_id", "author"]


def _sample(image_name: str) -> dict:
    return {"image_path": f"data/images/{image_name}"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """tmp_path 配下に data/images/・samples.db・attribution CSV を作り、モジュール定数を差し替える。"""
    images_dir = tmp_path / "data" / "images"
    images_dir.mkdir(parents=True)
    db_path = tmp_path / "data" / "samples.db"
    db_path.write_bytes(b"")
    attribution = tmp_path / "data" / "open_images_attribution.csv"

    monkeypatch.setattr(co, "ROOT", tmp_path)
    monkeypatch.setattr(co, "DB_PATH", db_path)
    monkeypatch.setattr(co, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(co, "ATTRIBUTION_CSV", attribution)

    db_updates: list[list[str]] = []
    monkeypatch.setattr(
        co._db, "delete_orphan_samples", lambda valid, path: db_updates.append(list(valid))
    )

    class Workspace:
        pass

    ws = Workspace()
    ws.images_dir = images_dir
    ws.attribution = attribution
    ws.db_updates = db_updates
    ws.set_samples = lambda samples: monkeypatch.setattr(
        co._db, "load_all_samples", lambda path: samples
    )
    return ws


def _make_images(images_dir: Path, names: list[str]) -> None:
    for name in names:
        (images_dir / name).write_bytes(b"jpg")


def _write_attribution(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ATTRIBUTION_HEADER)
        for image_id in ids:
            writer.writerow([image_id, "someone"])


def _attribution_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def _remaining(images_dir: Path) -> list[str]:
    return sorted(p.name for p in images_dir.iterdir())


class TestEmptyDatabaseGuard:

    def test_empty_db_deletes_no_image_and_returns_failure(self, workspace):
        names = ["a.jpg", "b.jpg", "c.png"]
        _make_images(workspace.images_dir, names)
        workspace.set_samples([])

        assert co.main(dry_run=False) != 0

        assert _remaining(workspace.images_dir) == sorted(names)

    def test_empty_db_leaves_attribution_csv_untouched(self, workspace):
        _make_images(workspace.images_dir, ["a.jpg"])
        _write_attribution(workspace.attribution, ["a", "b"])
        workspace.set_samples([])

        co.main(dry_run=False)

        assert _attribution_ids(workspace.attribution) == ["a", "b"]

    def test_empty_db_does_not_touch_the_database(self, workspace):
        _make_images(workspace.images_dir, ["a.jpg"])
        workspace.set_samples([])

        co.main(dry_run=False)

        assert workspace.db_updates == []


class TestOrphanRatioGuard:

    def test_db_referencing_no_existing_image_deletes_nothing(self, workspace):
        """別 DB の取り違え: DB の参照先が data/images/ に 1 枚も無い。"""
        names = ["a.jpg", "b.jpg", "c.jpg"]
        _make_images(workspace.images_dir, names)
        workspace.set_samples([_sample("other1.jpg"), _sample("other2.jpg")])

        assert co.main(dry_run=False) != 0

        assert _remaining(workspace.images_dir) == names
        assert workspace.db_updates == []

    def test_majority_orphans_deletes_nothing(self, workspace):
        names = [f"img{i}.jpg" for i in range(10)]
        _make_images(workspace.images_dir, names)
        workspace.set_samples([_sample("img0.jpg"), _sample("img1.jpg")])  # 8/10 が孤立

        assert co.main(dry_run=False) != 0

        assert _remaining(workspace.images_dir) == sorted(names)

    def test_abort_message_reports_counts_and_ratio(self, workspace, caplog):
        _make_images(workspace.images_dir, [f"img{i}.jpg" for i in range(10)])
        workspace.set_samples([_sample("img0.jpg"), _sample("img1.jpg")])

        co.main(dry_run=False)

        assert "8/10" in caplog.text
        assert "80%" in caplog.text
        assert "--force" in caplog.text

    def test_force_allows_deleting_a_majority_of_orphans(self, workspace):
        names = [f"img{i}.jpg" for i in range(10)]
        _make_images(workspace.images_dir, names)
        workspace.set_samples([_sample("img0.jpg"), _sample("img1.jpg")])

        assert co.main(dry_run=False, force=True) == 0

        assert _remaining(workspace.images_dir) == ["img0.jpg", "img1.jpg"]

    def test_force_does_not_override_the_empty_db_guard(self, workspace):
        _make_images(workspace.images_dir, ["a.jpg", "b.jpg"])
        workspace.set_samples([])

        assert co.main(dry_run=False, force=True) != 0

        assert _remaining(workspace.images_dir) == ["a.jpg", "b.jpg"]


class TestNormalCleanup:

    def test_only_orphan_images_are_deleted(self, workspace):
        _make_images(workspace.images_dir, ["keep1.jpg", "keep2.png", "orphan.jpg"])
        workspace.set_samples([_sample("keep1.jpg"), _sample("keep2.png")])

        assert co.main(dry_run=False) == 0

        assert _remaining(workspace.images_dir) == ["keep1.jpg", "keep2.png"]

    def test_orphan_attribution_rows_are_removed(self, workspace):
        _make_images(workspace.images_dir, ["keep1.jpg", "keep2.jpg", "orphan.jpg"])
        _write_attribution(workspace.attribution, ["keep1", "keep2", "orphan"])
        workspace.set_samples([_sample("keep1.jpg"), _sample("keep2.jpg")])

        co.main(dry_run=False)

        assert _attribution_ids(workspace.attribution) == ["keep1", "keep2"]

    def test_db_entries_without_image_are_reported_to_the_database(self, workspace):
        _make_images(workspace.images_dir, ["keep1.jpg", "keep2.jpg", "keep3.jpg"])
        workspace.set_samples(
            [_sample("keep1.jpg"), _sample("keep2.jpg"), _sample("keep3.jpg"), _sample("gone.jpg")]
        )

        co.main(dry_run=False)

        assert workspace.db_updates == [
            ["data/images/keep1.jpg", "data/images/keep2.jpg", "data/images/keep3.jpg"]
        ]

    def test_dry_run_deletes_nothing(self, workspace):
        names = ["keep1.jpg", "keep2.jpg", "orphan.jpg"]
        _make_images(workspace.images_dir, names)
        _write_attribution(workspace.attribution, ["keep1", "keep2", "orphan"])
        workspace.set_samples([_sample("keep1.jpg"), _sample("keep2.jpg")])

        assert co.main(dry_run=True) == 0

        assert _remaining(workspace.images_dir) == sorted(names)
        assert _attribution_ids(workspace.attribution) == ["keep1", "keep2", "orphan"]
        assert workspace.db_updates == []

    def test_no_orphans_is_a_clean_success(self, workspace):
        names = ["a.jpg", "b.jpg"]
        _make_images(workspace.images_dir, names)
        workspace.set_samples([_sample("a.jpg"), _sample("b.jpg")])

        assert co.main(dry_run=False) == 0

        assert _remaining(workspace.images_dir) == names


class TestMissingInputs:

    def test_missing_database_returns_failure(self, workspace, monkeypatch):
        monkeypatch.setattr(co, "DB_PATH", workspace.images_dir.parent / "absent.db")
        _make_images(workspace.images_dir, ["a.jpg"])

        assert co.main(dry_run=False) != 0

        assert _remaining(workspace.images_dir) == ["a.jpg"]
