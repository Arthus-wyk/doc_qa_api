from __future__ import annotations

from app.ingestion.models import DocumentMeta
from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.pdf_parser import PDFStructuredParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.csv_parser import CSVTableParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.fallback_parser import FallbackParser


class ParserRouter:
    # 解析器注册中心。顺序会影响匹配结果。
    def __init__(self) -> None:
        # 具体解析器放在前面，通用兜底解析器放在最后。
        self.parsers: list[BaseParser] = [
            PDFStructuredParser(),
            MarkdownParser(),
            CSVTableParser(),
            TextParser(),
            FallbackParser(),
        ]

    def pick_parser(self, meta: DocumentMeta) -> BaseParser:
        # 返回第一个声明支持该文件的解析器。
        for parser in self.parsers:
            if parser.supports(meta):
                print("parser:",parser.name)
                return parser
        # 防御性兜底；通常不会走到这里，因为 FallbackParser.supports 恒为 True。
        return FallbackParser()

