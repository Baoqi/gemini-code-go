# 环境变量配置

## 核心配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `GEMINI_API_KEY` | (必需) | Google Gemini API 密钥 |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API 基础 URL |
| `BIG_MODEL` | `gemini-2.5-flash` | 大模型映射（用于 sonnet 类请求） |
| `SMALL_MODEL` | `gemini-2.0-flash` | 小模型映射（用于 haiku 类请求） |
| `PORT` | `8082` | 服务器监听端口 |

## 会话管理

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SESSION_TIMEOUT` | `10m` | 会话超时时间（支持时间单位如 5m, 1h） |

## 重试与容错

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `GEMINI_INTERNAL_ERROR_INITIAL_DELAY` | `30s` | 内部错误初始重试延迟 |
| `GEMINI_INTERNAL_ERROR_MAX_RETRIES` | `2` | 内部错误最大重试次数 |
| `GEMINI_RATE_LIMIT_DEFAULT_DELAY` | `60s` | 限流默认延迟时间 |
| `GEMINI_RATE_LIMIT_MAX_RETRIES` | `2` | 限流最大重试次数 |

## 日志控制

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_MESSAGES_HEADERS` | 空 | 设置为 `1` 启用请求头日志记录 |
| `LOG_MESSAGES_BODY` | 空 | 设置为 `1` 启用请求体日志记录 |
| `LOG_FULL_OUTPUT` | 空 | 设置为 `1` 在 Langfuse 中记录完整的原始 Gemini 响应 |
| `SKIP_STREAM_OUTPUT` | `1` | 设置为 `0` 恢复流式请求的 Generation 记录到 Langfuse（默认跳过） |

## Langfuse 集成

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `GEMINI_CODE_ENABLE_LANGFUSE` | 空 | **必需**：设置为任意非空值以启用 Langfuse 功能 |
| `LANGFUSE_PUBLIC_KEY` | 空 | Langfuse 公钥（启用追踪需要） |
| `LANGFUSE_SECRET_KEY` | 空 | Langfuse 密钥（启用追踪需要） |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse 服务器地址 |
| `LANGFUSE_DEBUG` | 空 | 设置为 `1` 启用 Langfuse 调试日志 |
| `LANGFUSE_RELEASE` | 空 | 发布版本标签 |
| `LANGFUSE_ENVIRONMENT` | `production` | 环境标签 |

## 使用示例

### 基础启动（不启用 Langfuse）
```bash
export GEMINI_API_KEY="your-api-key"
./gemini-code-go
```

### 启用 Langfuse 追踪
```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_CODE_ENABLE_LANGFUSE=1
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
./gemini-code-go
```

### 开发调试模式（启用 Langfuse）
```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_CODE_ENABLE_LANGFUSE=1
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export LOG_MESSAGES_HEADERS=1
export LOG_MESSAGES_BODY=1
export LOG_FULL_OUTPUT=1
export LANGFUSE_DEBUG=1
./gemini-code-go
```

### 生产环境（默认精简记录）
```bash
export GEMINI_API_KEY="your-api-key"
export SESSION_TIMEOUT=5m
./gemini-code-go
```

### 开发环境（完整记录，启用 Langfuse）
```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_CODE_ENABLE_LANGFUSE=1
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export SKIP_STREAM_OUTPUT=0
export LOG_MESSAGES_HEADERS=1
./gemini-code-go
```

### 高负载环境（增强重试）
```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_RATE_LIMIT_MAX_RETRIES=5
export GEMINI_RATE_LIMIT_DEFAULT_DELAY=30s
export GEMINI_INTERNAL_ERROR_MAX_RETRIES=3
./gemini-code-go
```

## 关于 SKIP_STREAM_OUTPUT

默认情况下，系统会跳过为流式请求创建 Generation 记录。可以通过设置 `SKIP_STREAM_OUTPUT=0` 来恢复记录这些 Generation。

**默认行为（SKIP_STREAM_OUTPUT=1 或未设置）：**
```
Session Trace
├── Span: request_1 (无 Generation 子记录)
└── Span: request_2
    └── Generation: gemini_generate (实际内容)
```

**恢复流式记录模式（SKIP_STREAM_OUTPUT=0）：**
```
Session Trace
├── Span: request_1
│   └── Generation: gemini_generate (包含流式状态信息)
└── Span: request_2  
    └── Generation: gemini_generate (实际内容)
```

这样做的好处：

1. **减少记录数量** - 避免创建内容价值较低的 Generation 记录
2. **节省存储空间** - 不记录流式状态信息
3. **简化分析** - Langfuse 中只显示真正有内容价值的 Generation
4. **保持完整性** - 非流式请求仍会正常记录所有信息

注意：Span 记录仍会保留，以维护请求的完整追踪链路。 