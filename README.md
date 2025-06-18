# Gemini Code – Go Edition

本项目是 Gemini Code 的 Go 语言重写版本，实现了与 Anthropic Messages API 兼容的代理服务器，能够将 Claude/Anthropic 风格的请求转换为 Google Gemini 的 API 调用。

## 主要特点

1. **零外部框架**：仅依赖 Go 标准库 (`net/http`, `encoding/json` 等)。
2. **环境变量配置**：
   * `GEMINI_API_KEY`  (必填) – Google AI Studio / Vertex AI 的 API Key。
   * `BIG_MODEL`       (可选) – 默认为 `gemini-2.5-flash-preview-05-20`。
   * `SMALL_MODEL`     (可选) – 默认为 `gemini-2.0-flash`。
   * `PORT`            (可选) – 监听端口，默认 `8082`。
   * `GEMINI_BASE_URL` (可选) – Gemini REST Base URL，默认 `https://generativelanguage.googleapis.com/v1beta`。
3. **兼容的路由**
   * `POST /v1/messages` – 发送对话消息，支持 `stream=true` 流式响应。
   * `POST /v1/messages/count_tokens` – 计算 Token 数量（调用 Gemini `/countTokens` 接口）。
   * `GET  /` – 根路径，返回欢迎信息。
   * `GET  /sysmon/health/{liveness|readiness}` – Kubernetes 健康检查。
4. **简单模型映射** – `sonnet` → BIG_MODEL，`haiku` → SMALL_MODEL，其余按请求字段或默认 BIG_MODEL。
5. **日志** – 直接使用 Go 标准库 `log`。

## 快速开始

```bash
# 克隆并进入目录
cd gemini-code-go

# 设置环境变量
export GEMINI_API_KEY="<YOUR_API_KEY>"

# 构建
go build -o gemini-server

# 运行
./gemini-server   # 默认 0.0.0.0:8082
```

## 对比 Python 版本
* Python 版使用 **FastAPI + LiteLLM**，功能更完整；Go 版强调**低依赖、高性能、易部署**。
* 当前 Go 版仅实现最常见的文本对话场景，复杂的工具调用、图像输入等高级特性将在后续迭代补充。

## 生产部署建议
1. 使用 `systemd` 或 `supervisor` 保持进程常驻。
2. 建议在边缘层（如 Nginx）对 `/v1/messages` 启用 WebSocket/SSE 反向代理支持。
3. 根据业务情况设置 `uLimits` 与连接池，避免大量并发导致的 FD 耗尽。

---
**MIT License** – © 2024 