# Document QA API

一个面向真实文档场景的 RAG 后端项目。  
目标是把“上传文档 -> 建索引 -> 语义问答”做成一条稳定、可扩展、可演进的工程链路，而不只是一个 Demo。

## 项目定位

这个项目关注的是“文档问答系统的工程化实现”，核心不是单点模型效果，而是完整系统能力：

- 多源文档解析与统一表示
- 检索链路的可组合设计（向量检索 + BM25）
- 检索后重排序提升相关性
- 基于状态图的问答流程编排
- 标准化 API 响应与异常处理

## 我做了什么

我把项目拆成了清晰的模块边界，并分别处理了文档接入、检索、问答编排、接口层与可运维性：

- 设计并实现 `ingestion` 管道：解析、分块、过滤、入库
- 实现混合检索策略：LlamaIndex 向量召回 + BM25 关键词召回
- 接入 Cohere Rerank 做二次排序，提升候选片段质量
- 使用 LangGraph 将问答流程显式建模，方便后续扩展节点
- 建立统一返回结构与全局异常处理，保证 API 行为一致

## 亮点设计

### 1) 解析层具备容错能力

优先走 Unstructured 解析复杂格式；若 PDF 解析失败，自动回退到 `pypdf`，避免链路中断。

### 2) 分块策略不是单一切片

项目不是固定 `chunk_size` 的粗暴切分，而是按文档结构选择分块器：

- `SectionChunker`
- `ParagraphChunker`
- `TableChunker`
- `RecursiveFallbackChunker`

### 3) 混合检索提升召回鲁棒性

向量检索擅长语义相似，BM25 擅长关键词命中。两者融合后，再去重与重排序，在不同类型问题上更稳定。

### 4) 问答流程可观察、可改造

通过 LangGraph 明确建模：

`retrieve_docs -> generate_answer -> package_response`

流程化的好处是后续可以自然插入新节点，比如查询改写、引用约束、答案评估等。

## 技术选型

- **FastAPI**：高性能 API 框架，天然适合服务化
- **LangGraph**：显式状态流转，优于“黑盒链式调用”
- **LlamaIndex + Chroma**：向量检索基础设施
- **BM25**：补足关键词召回能力
- **Cohere Rerank**：提升最终上下文质量
- **Ollama**：本地模型部署灵活，便于开发迭代

## 系统架构

```text
Upload Document
  -> Parser Router
  -> Chunker Strategy
  -> Vector Store (Chroma)
  -> Chunk Store (JSONL for BM25)

Ask Question
  -> Query Rewrite
  -> Hybrid Retrieve (Vector + BM25)
  -> Rerank
  -> LLM Generate
  -> Structured Response
```

## 工程化细节

- 统一响应模型：`success / message / data / error / meta`
- 全局异常处理：业务异常、HTTP 异常、参数校验异常统一出口
- 启动时初始化：LLM、向量库、检索器图状态一次就绪
- 文档来源追踪：保留 `source_file`、`chunk_id` 等元数据用于溯源

## 项目结构

```text
app/
├─ api/routes/        # 对外 API 层
├─ core/              # 配置、异常、图编排、通用响应
├─ ingestion/         # 解析与分块主流程
├─ rag/               # 检索、重排、向量存储
├─ services/          # 业务服务与检索器刷新
└─ schemas/           # 请求/响应模型
```

## 当前状态

项目已完成从文档入库到问答返回的主链路，具备可演示与继续迭代的基础。  
下一阶段重点会放在：

- 配置收敛（去 hardcode，统一配置驱动）
- 依赖清单固定（`requirements.txt` / `pyproject.toml`）
- 自动化测试补齐（重点覆盖解析与问答核心路径）
- 鉴权与限流（走向生产化）

## 快速预览

本项目提供以下接口：

- `GET /health`
- `POST /documents/upload`
- `POST /documents/upload-and-ingest`
- `GET /documents/`
- `POST /qa/ask`

如果你想快速体验，可启动服务后访问：`/docs`。
