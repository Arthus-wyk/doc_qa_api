from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.analyzer import analyze_file
from app.ingestion.router import ParserRouter
from app.ingestion.quality import filter_chunks
from app.ingestion.chunkers.section_chunker import SectionChunker
from app.ingestion.chunkers.paragraph_chunker import ParagraphChunker
from app.ingestion.chunkers.table_chunker import TableChunker
from app.ingestion.chunkers.recursive_chunker import RecursiveFallbackChunker
from app.rag.vectorstore import get_vectorstore
from app.rag.llamaindex_store import get_embed_model, get_storage_context
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class IngestionService:
    # 完整入库流水线协调器：
    # 文件分析 -> 解析器选择 -> 解析 -> 切分 -> 质量过滤 -> 向量库写入。
    def __init__(self) -> None:
        # 解析器路由器根据文件元数据（扩展名、MIME 等）选择解析器。
        self.parser_router = ParserRouter()
        # 切分器按优先级排列：结构化策略优先，兜底策略最后。
        self.chunkers = [
            SectionChunker(),
            ParagraphChunker(),
            TableChunker(),
            RecursiveFallbackChunker(),
        ]

    def ingest(self, file_path: Path) -> dict:
        # 第 1 步：推断文件元数据，用于解析路由与下游追踪。
        meta = analyze_file(file_path)
        # 第 2 步：为当前文档类型选择解析器实现。
        parser = self.parser_router.pick_parser(meta)

        # 第 3 步：将原始文件内容标准化为 ParsedDocument（elements + meta）。
        parsed_doc = parser.parse(file_path, meta)

        # 第 4 步：为当前解析结构选择最合适的切分策略。
        chunker = self._pick_chunker(parsed_doc)
        # 第 5 步：将解析元素转换为可向量化的 chunks。
        units = chunker.chunk(parsed_doc)
        # 第 6 步：移除明显低价值 chunks，降低索引噪声。
        # units = filter_chunks(units)
        if units:
            try:
                # 设置 LlamaIndex 的 embedding 模型
                Settings.embed_model = get_embed_model()

                # 你的 unit -> LlamaIndex TextNode
                nodes = []
                for i, unit in enumerate(units):
                    node_id = self._make_node_id(parsed_doc.meta.source_file, i, unit.metadata)

                    nodes.append(
                        TextNode(
                            text=unit.text,
                            metadata=unit.metadata or {},
                            id_=node_id,
                        )
                    )

                storage_context = get_storage_context()

                # 写入 Chroma
                VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )

            except Exception:
                logger.exception(
                    "Vectorstore write failed for file: %s",
                    parsed_doc.meta.source_file,
                )

        return {
            "source_file": parsed_doc.meta.source_file,
            "parser": parsed_doc.meta.parser_name,
            "chunk_strategy": chunker.name,
            "elements_count": len(parsed_doc.elements),
            "chunks_count": len(units),
        }

    def _pick_chunker(self, parsed_doc):
        # 按顺序选择第一个支持的切分器；优先级由 __init__ 中顺序决定。
        for chunker in self.chunkers:
            if chunker.supports(parsed_doc):
                print("chunker:",chunker.name)
                return chunker
        # 防御性兜底：该路径通常不会触发，因为 recursive chunker 支持所有文档。
        return RecursiveFallbackChunker()
    def _make_node_id(self, source_file: str, index: int, metadata: dict | None) -> str:
        page = None
        if metadata:
            page = metadata.get("page") or metadata.get("page_number")
        page_part = f"p{page}_" if page is not None else ""
        return f"{Path(source_file).name}:{page_part}{index}"
