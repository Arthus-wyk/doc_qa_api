from __future__ import annotations

import mimetypes
from pathlib import Path

from app.ingestion.models import DocumentMeta


def analyze_file(file_path: Path) -> DocumentMeta:
    # 先读取轻量级文件信息，用于路由选择和可观测性。
    stat = file_path.stat()
    # MIME 类型是尽力猜测，未知类型时可能为 None。
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return DocumentMeta(
        source_file=file_path.name,
        file_ext=file_path.suffix.lower(),
        mime_type=mime_type,
        file_size=stat.st_size,
    )

