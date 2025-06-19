# Gemini Code – Go Edition

本项目是 [coffeegrind123/gemini-code](https://github.com/coffeegrind123/gemini-code) 的 Go 语言重写版本，原项目使用 Python + FastAPI 实现。借助 Cursor AI 的帮助，我们将其翻译为 Go 语言，实现了与 Anthropic Messages API 兼容的代理服务器，能够将 Claude/Anthropic 风格的请求转换为 Google Gemini 的 API 调用。

## 主要特点

1. **零外部框架**：仅依赖 Go 标准库 (`net/http`, `encoding/json` 等)。
2. **会话管理**：支持基于 API Key + User-Agent 的会话关联，避免相关请求在追踪中独立显示。
3. **Langfuse 集成**：可选的请求追踪和监控支持，便于分析和调试。
4. **环境变量配置**：
   * `GEMINI_API_KEY`  (必填) – Google AI Studio / Vertex AI 的 API Key。
   * `BIG_MODEL`       (可选) – 默认为 `gemini-2.5-flash`。
   * `SMALL_MODEL`     (可选) – 默认为 `gemini-2.0-flash`。
   * `PORT`            (可选) – 监听端口，默认 `8082`。
   * `GEMINI_BASE_URL` (可选) – Gemini REST Base URL，默认 `https://generativelanguage.googleapis.com/v1beta`。
5. **兼容的路由**
   * `POST /v1/messages` – 发送对话消息，支持 `stream=true` 流式响应。
   * `POST /v1/messages/count_tokens` – 计算 Token 数量（调用 Gemini `/countTokens` 接口）。
   * `GET  /` – 根路径，返回欢迎信息。
   * `GET  /sysmon/health/{liveness|readiness}` – Kubernetes 健康检查。
6. **简单模型映射** – `sonnet` → BIG_MODEL，`haiku` → SMALL_MODEL，其余按请求字段或默认 BIG_MODEL。
7. **日志** – 直接使用 Go 标准库 `log`。

## 快速开始

```bash
# 克隆并进入目录
cd gemini-code-go

# 设置环境变量
export GEMINI_API_KEY="<YOUR_API_KEY>"

# 构建
go build -o gemini-code-go

# 运行
./gemini-code-go   # 默认 0.0.0.0:8082
```

## Langfuse 集成（可选）

本项目支持 [Langfuse](https://langfuse.com/) 集成，用于请求追踪、监控和分析。

### 启用 Langfuse

要启用 Langfuse 功能，需要设置以下环境变量：

```bash
# 必需：启用 Langfuse 功能的总开关
export GEMINI_CODE_ENABLE_LANGFUSE=1

# 必需：Langfuse 认证信息
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

# 可选：Langfuse 服务器地址（默认为 cloud.langfuse.com）
export LANGFUSE_HOST="https://cloud.langfuse.com"

# 可选：环境标签
export LANGFUSE_ENVIRONMENT="production"
export LANGFUSE_RELEASE="v1.0.0"
```

### 功能特点

- **会话关联**：相关请求会在同一个 Session Trace 中显示，避免独立的记录
- **智能过滤**：默认跳过流式请求的 Generation 记录，只保留有内容价值的记录（可通过 `SKIP_STREAM_OUTPUT=0` 恢复）
- **调试支持**：支持完整请求/响应记录和调试模式

### 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GEMINI_CODE_ENABLE_LANGFUSE` | 空 | **必需**：设置为任意非空值以启用 Langfuse |
| `SKIP_STREAM_OUTPUT` | `1` | 设置为 `0` 恢复流式请求的 Generation 记录 |
| `LOG_FULL_OUTPUT` | 空 | 设置为 `1` 记录完整的原始 Gemini 响应 |
| `LANGFUSE_DEBUG` | 空 | 设置为 `1` 启用调试日志 |

完整的环境变量配置请参考 [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)。

## 对比 Python 版本
* Python 版使用 **FastAPI + LiteLLM**，功能更完整；Go 版强调**低依赖、高性能、易部署**。
* 当前 Go 版仅实现最常见的文本对话场景，复杂的工具调用、图像输入等高级特性将在后续迭代补充。

## 生产部署建议
1. 使用 `systemd` 或 `supervisor` 保持进程常驻。
2. 建议在边缘层（如 Nginx）对 `/v1/messages` 启用 WebSocket/SSE 反向代理支持。
3. 根据业务情况设置 `uLimits` 与连接池，避免大量并发导致的 FD 耗尽。

## 致谢

- 感谢 [coffeegrind123/gemini-code](https://github.com/coffeegrind123/gemini-code) 提供的原始 Python 实现

---
**MIT License** – © 2025
