import os
os.environ["HF_HOME"] = r"D:\HuggingFace_Cache"
import re
from pathlib import Path
from typing import Any
from app.ingestion.models import ParsedDocument, ParsedElement
# ==========================================
# 1. 替换为新版 marker-pdf 的导入方式
# ==========================================
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from app.ingestion.parsers.base import BaseParser

# 将原来的 _MODEL_LST 改为缓存 artifact_dict
_ARTIFACT_DICT: Any | None = None

def _get_artifact_dict() -> Any:
    global _ARTIFACT_DICT
    if _ARTIFACT_DICT is None:
        # 新版加载所有模型字典的方法
        _ARTIFACT_DICT = create_model_dict()
    return _ARTIFACT_DICT
            
# 假设这些是你原本定义的基类和数据结构
# from your_module import BaseParser, ParsedDocument, ParsedElement, DocumentMeta

class AcademicPDFParser(BaseParser):
    """
    专门针对学术论文的解析器。
    底层使用 Marker 视觉模型提取排版、公式和表格，输出 Markdown。
    然后将 Markdown 解析为系统标准的 ParsedElement 列表。
    """
    name = "academic_pdf_marker"

    def supports(self, meta: 'DocumentMeta') -> bool:
        # 仅针对 PDF 启用此高级解析器
        return meta.file_ext.lower() == ".pdf"

    def parse(self, file_path: Path, meta: 'DocumentMeta') -> 'ParsedDocument':
        meta.parser_name = self.name
        
        try:
            # 1. 调用底层的学术模型，将 PDF 转为原生 Markdown
            md_text, md_metadata = self._convert_pdf_to_md_with_marker(file_path)
            
            # 将模型提取的元数据（如语言、总页数等）存入你系统的 meta 中
            meta.extra["marker_metadata"] = md_metadata
            
            # 2. 将 Markdown 文本结构化为你现有的 Element 对象
            elements = self._parse_markdown_to_elements(md_text)
            
            return ParsedDocument(meta=meta, elements=elements)
            
        except Exception as exc:
            # 降级容灾：如果视觉模型显存不足或失败，记录错误并降级
            meta.extra["academic_parser_error"] = str(exc)
            
            # 可以在这里调用你之前的 _parse_pdf_with_fallback
            fallback_text = file_path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                meta=meta,
                elements=[ParsedElement(element_type="unknown", text=fallback_text[:50000])],
            )

    # ==========================================
    # 2. 重写 Marker 调用逻辑以适配新版本 API
    # ==========================================
    def _convert_pdf_to_md_with_marker(self, file_path: Path) -> tuple[str, dict]:
        """
        调用新版 marker-pdf 模型将 PDF 转换为 Markdown。
        """
        try:
            # 1. 基础配置
            config = {
                "output_format": "markdown"
            }
            config_parser = ConfigParser(config)
            
            # 2. 获取全局缓存的模型字典
            artifact_dict = _get_artifact_dict()
            
            # 3. 初始化转换器
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=artifact_dict,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer()
            )
            
            # 4. 渲染文档
            rendered = converter(str(file_path))
            
            # 5. 提取文本内容和元数据
            text, out_meta, images = text_from_rendered(rendered)
            
            return text, out_meta
            
        except ImportError:
            raise ImportError(
                "缺少 marker-pdf 依赖。请运行 `pip install marker-pdf` 并确保安装了 PyTorch。"
            )

    def _parse_markdown_to_elements(self, md_text: str) -> list['ParsedElement']:
        """
        将整篇 Markdown 按块（Block）拆解，并映射为你现有的 Element 类型。
        这样你的 SectionChunker 就能完美识别出 'title'、'table' 和 'narrative_text'。
        """
        elements: list['ParsedElement'] = []
        
        # 按空行对 Markdown 进行块级别分割
        blocks = re.split(r'\n{2,}', md_text.strip())
        
        current_section = "UNTITLED"
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # 1. 识别 Markdown 标题 (e.g., "## 1. Introduction")
            header_match = re.match(r'^(#{1,6})\s+(.+)$', block)
            if header_match:
                level = len(header_match.group(1))
                title_text = header_match.group(2).strip()
                current_section = title_text  # 更新当前所属章节
                
                elements.append(
                    ParsedElement(
                        element_type="title",
                        text=title_text,
                        section_hint=current_section,
                        metadata={"md_level": level, "is_toc_like": False}
                    )
                )
                continue

            # 2. 识别表格块 (Markdown table)
            if block.startswith('|') and '-|-' in block:
                elements.append(
                    ParsedElement(
                        element_type="table",
                        text=block,
                        section_hint=current_section,
                        metadata={"format": "markdown_table"}
                    )
                )
                continue
                
            # 3. 识别独立公式块 (LaTeX equations: $$ E=mc^2 $$)
            if block.startswith('$$') and block.endswith('$$'):
                elements.append(
                    ParsedElement(
                        element_type="math",  # 建议在你的系统中新增此类型
                        text=block,
                        section_hint=current_section,
                        metadata={"format": "latex"}
                    )
                )
                continue
                
            # 4. 识别代码块
            if block.startswith('```') and block.endswith('```'):
                elements.append(
                    ParsedElement(
                        element_type="code",
                        text=block,
                        section_hint=current_section,
                    )
                )
                continue

            # 5. 常规正文段落
            elements.append(
                ParsedElement(
                    element_type="narrative_text",
                    text=block,
                    section_hint=current_section,
                )
            )

        return elements