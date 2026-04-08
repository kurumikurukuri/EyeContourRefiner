"""exp_utils.py: train_expN.py 共通ユーティリティ。

- run_with_log    : サブプロセスを実行し stdout/stderr をログファイルと端末に同時出力
- find_model_dir  : outputs/ 以下で fold モデルを含むディレクトリを探す
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_with_log(cmd: list, log_path: Path, cwd: Path | None = None) -> int:
    """コマンドを実行し、stdout/stderr を log_path に追記しつつ端末にも表示する。

    Args:
        cmd:      実行するコマンドのリスト
        log_path: ログ追記先ファイル（親ディレクトリは自動作成）
        cwd:      サブプロセスの作業ディレクトリ（None の場合は呼び出し元のまま）

    Returns:
        プロセスの returncode
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd is not None else None,
    )
    with open(log_path, "a", encoding="utf-8") as log_f:
        for line in process.stdout:
            sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
            log_f.write(line)
    process.wait()
    return process.returncode


def find_model_dir(output_dir: Path) -> Path:
    """output_dir 直下で eyeliner_detection_fold*.pt を含む最新ディレクトリを返す。

    train.py は output_dir/{exp_name}/ にモデルを保存するため、
    その exp_name ディレクトリを動的に探す。

    Raises:
        FileNotFoundError: モデルが見つからない場合
    """
    candidates = [
        d for d in output_dir.iterdir()
        if d.is_dir() and list(d.glob("eyeliner_detection_fold*.pt"))
    ]
    if not candidates:
        raise FileNotFoundError(f"訓練済みモデルが見つかりません: {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)
