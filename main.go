package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ----- Configuration -----

var (
	// Defaults specified by user rules
	defaultBigModel   = getEnv("BIG_MODEL", "gemini-2.5-flash")
	defaultSmallModel = getEnv("SMALL_MODEL", "gemini-2.0-flash")

	geminiAPIKey  = os.Getenv("GEMINI_API_KEY")
	geminiBaseURL = getEnv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

	// Whether to log /v1/messages request body (controlled via env)
	logMessagesBody = getEnv("LOG_MESSAGES_BODY", "") != ""
	// Whether to log /v1/messages request headers (controlled via env)
	logMessagesHeaders = getEnv("LOG_MESSAGES_HEADERS", "") != ""
	// Whether to log full raw output in Langfuse (controlled via env)
	logFullOutput = getEnv("LOG_FULL_OUTPUT", "") != ""

	// Global backoff & retry configuration
	backoffMutex sync.Mutex
	backoffUntil time.Time

	internalErrorInitialDelay = getEnvDuration("GEMINI_INTERNAL_ERROR_INITIAL_DELAY", 30*time.Second)
	internalErrorMaxRetries   = getEnvInt("GEMINI_INTERNAL_ERROR_MAX_RETRIES", 2)
	rateLimitDefaultDelay     = getEnvDuration("GEMINI_RATE_LIMIT_DEFAULT_DELAY", 60*time.Second)
	rateLimitMaxRetries       = getEnvInt("GEMINI_RATE_LIMIT_MAX_RETRIES", 2)

	// Atomic counter for incoming requests – used to correlate logs.
	reqIDCounter uint64

	// Session management
	sessionManager *SessionManager
)

// ----- Session Management -----

type SessionInfo struct {
	TraceID       string
	LastActivity  time.Time
	RequestCount  int
	TotalMessages int
	TotalTools    int
	UserAgent     string
	ClientInfo    string
	mutex         sync.RWMutex
}

type SessionManager struct {
	sessions    map[string]*SessionInfo
	mutex       sync.RWMutex
	timeout     time.Duration
	cleanupTick *time.Ticker
}

func NewSessionManager() *SessionManager {
	timeout := getEnvDuration("SESSION_TIMEOUT", 10*time.Minute) // 10 minutes default
	if timeout < time.Minute {
		timeout = time.Minute // minimum 1 minute
	}

	sm := &SessionManager{
		sessions: make(map[string]*SessionInfo),
		timeout:  timeout,
	}

	// Start cleanup goroutine
	sm.cleanupTick = time.NewTicker(1 * time.Minute)
	go sm.cleanupExpiredSessions()

	return sm
}

func (sm *SessionManager) getSessionKey(r *http.Request) string {
	// Use API key + User-Agent + client info to create session key
	apiKey := r.Header.Get("X-Api-Key")
	userAgent := r.Header.Get("User-Agent")

	// Create a hash for session identification
	h := sha256.New()
	h.Write([]byte(apiKey))
	h.Write([]byte(userAgent))

	// Add timing component for session grouping
	timeoutSeconds := int64(sm.timeout.Seconds())
	if timeoutSeconds <= 0 {
		timeoutSeconds = 600 // fallback to 10 minutes
	}
	timeWindow := time.Now().Unix() / timeoutSeconds
	h.Write([]byte(fmt.Sprintf("%d", timeWindow)))

	return hex.EncodeToString(h.Sum(nil))[:16] // Use first 16 chars
}

func (sm *SessionManager) getOrCreateSession(r *http.Request, model string, toolCount, messageCount int) (*SessionInfo, bool) {
	sessionKey := sm.getSessionKey(r)

	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	session, exists := sm.sessions[sessionKey]
	if !exists {
		// Create new session
		userAgent := r.Header.Get("User-Agent")
		clientInfo := extractClientInfo(r)

		traceID := lfStartTrace("claude_session", map[string]interface{}{
			"session_key": sessionKey,
			"user_agent":  userAgent,
			"client_info": clientInfo,
			"model":       model,
			"started_at":  time.Now().UTC().Format(time.RFC3339),
		})

		session = &SessionInfo{
			TraceID:       traceID,
			LastActivity:  time.Now(),
			RequestCount:  0,
			TotalMessages: 0,
			TotalTools:    0,
			UserAgent:     userAgent,
			ClientInfo:    clientInfo,
		}
		sm.sessions[sessionKey] = session

		log.Printf("[SESSION] New session created: %s (trace: %s)", sessionKey, traceID)
	}

	// Update session stats
	session.mutex.Lock()
	session.LastActivity = time.Now()
	session.RequestCount++
	session.TotalMessages += messageCount
	session.TotalTools += toolCount
	session.mutex.Unlock()

	return session, !exists
}

func (sm *SessionManager) cleanupExpiredSessions() {
	for {
		select {
		case <-sm.cleanupTick.C:
			sm.cleanupExpired()
		}
	}
}

func (sm *SessionManager) cleanupExpired() {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()

	cutoff := time.Now().Add(-sm.timeout - time.Minute) // Add 1 minute buffer
	var expiredKeys []string

	for key, session := range sm.sessions {
		session.mutex.RLock()
		if session.LastActivity.Before(cutoff) {
			expiredKeys = append(expiredKeys, key)

			// Finish the session trace
			lfFinishTrace(session.TraceID, map[string]interface{}{
				"session_completed": true,
				"total_requests":    session.RequestCount,
				"total_messages":    session.TotalMessages,
				"total_tools":       session.TotalTools,
				"duration_seconds":  time.Since(session.LastActivity.Add(-sm.timeout)).Seconds(),
			}, nil)

			log.Printf("[SESSION] Session expired: %s (requests: %d)", key, session.RequestCount)
		}
		session.mutex.RUnlock()
	}

	for _, key := range expiredKeys {
		delete(sm.sessions, key)
	}
}

func extractClientInfo(r *http.Request) string {
	parts := []string{}

	if app := r.Header.Get("X-App"); app != "" {
		parts = append(parts, "app:"+app)
	}
	if version := r.Header.Get("X-Stainless-Package-Version"); version != "" {
		parts = append(parts, "v:"+version)
	}
	if os := r.Header.Get("X-Stainless-Os"); os != "" {
		parts = append(parts, "os:"+os)
	}
	if runtime := r.Header.Get("X-Stainless-Runtime"); runtime != "" {
		parts = append(parts, "rt:"+runtime)
	}

	if len(parts) == 0 {
		return "unknown"
	}
	return strings.Join(parts, ",")
}

const defaultGeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta"

func init() {
	if geminiAPIKey == "" {
		log.Println("[WARN] GEMINI_API_KEY not set – the server will start but upstream calls will fail.")
	}
	// initialize Langfuse if env vars present
	initLangfuse()
	// initialize session manager
	sessionManager = NewSessionManager()
}

// getEnv returns value from environment or fallback if empty.
func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvDuration(key string, d time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if p, err := time.ParseDuration(v + "s"); err == nil {
			return p
		}
	}
	return d
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

// ----- Request / Response structures (partial subset) -----

type FunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type Part struct {
	Text         string        `json:"text,omitempty"`
	FunctionCall *FunctionCall `json:"functionCall,omitempty"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts,omitempty"`
}

// GenerationConfig groups tuning parameters for Gemini.
type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type GenerationRequest struct {
	Contents         []Content         `json:"contents"`
	Tools            []Tool            `json:"tools,omitempty"`
	ToolConfig       *ToolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig *GenerationConfig `json:"generationConfig,omitempty"`
}

type TokenCountRequestUpstream struct {
	Contents []Content `json:"contents"`
}

type TokenCountResponseUpstream struct {
	TotalTokens int `json:"totalTokens"`
}

// ----- Anthropic-compatible structures (subset) -----

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type MessagesRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Messages      []AnthropicMessage `json:"messages"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Tools         []ToolDescriptor   `json:"tools,omitempty"`
	ToolChoice    *ToolChoice        `json:"tool_choice,omitempty"`
}

type MessagesResponse struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// TokenCountRequest mirrors Anthropic route

type TokenCountRequest struct {
	Model    string             `json:"model"`
	Messages []AnthropicMessage `json:"messages"`
}

// TokenCountResponse mirrors Anthropic route

type TokenCountResponse struct {
	InputTokens int `json:"input_tokens"`
}

// ----- Helpers -----

// mapRequestedModel maps Claude-style names to Gemini ones based on heuristics.
func mapRequestedModel(requested string) string {
	l := strings.ToLower(requested)

	// Allow fully-qualified gemini/<model>
	if strings.HasPrefix(l, "gemini/") {
		return strings.TrimPrefix(requested, "gemini/")
	}

	haikuRegex := regexp.MustCompile(`haiku`) // capture haiku variations
	sonnetRegex := regexp.MustCompile(`sonnet`)

	switch {
	case haikuRegex.MatchString(l):
		return defaultSmallModel
	case sonnetRegex.MatchString(l):
		return defaultBigModel
	default:
		// If the caller already passed a valid Gemini model, keep as-is.
		if strings.HasPrefix(l, "gemini-") {
			return requested
		}
		// fallback
		return defaultBigModel
	}
}

// convertToGemini prepares GenerationRequest for upstream.
func convertToGemini(req *MessagesRequest, geminiModel string) (*GenerationRequest, error) {
	contents := make([]Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		switch v := m.Content.(type) {
		case string:
			contents = append(contents, Content{
				Role:  mapRole(m.Role),
				Parts: []Part{{Text: v}},
			})
		case []interface{}:
			// If content is array of text blocks etc – flatten text for now.
			var combined strings.Builder
			for _, part := range v {
				// we attempt to marshal part and append.
				b, _ := json.Marshal(part)
				combined.Write(b)
			}
			contents = append(contents, Content{Role: mapRole(m.Role), Parts: []Part{{Text: combined.String()}}})
		default:
			// Attempt to stringify
			b, _ := json.Marshal(v)
			contents = append(contents, Content{Role: mapRole(m.Role), Parts: []Part{{Text: string(b)}}})
		}
	}

	genCfg := &GenerationConfig{MaxOutputTokens: req.MaxTokens}
	if req.Temperature != nil {
		genCfg.Temperature = req.Temperature
	}
	if req.TopP != nil {
		genCfg.TopP = req.TopP
	}
	if req.TopK != nil {
		genCfg.TopK = req.TopK
	}
	if len(req.StopSequences) > 0 {
		genCfg.StopSequences = req.StopSequences
	}

	gReq := &GenerationRequest{
		Contents:         contents,
		GenerationConfig: genCfg,
	}

	// Tools conversion
	if len(req.Tools) > 0 {
		var fns []FunctionDeclaration
		for _, td := range req.Tools {
			cleaned := cleanGeminiSchema(td.InputSchema)
			fns = append(fns, FunctionDeclaration{
				Name:        td.Name,
				Description: td.Description,
				Parameters:  cleaned,
			})
		}
		gReq.Tools = []Tool{{FunctionDeclarations: fns}}
	}

	// tool_choice => toolConfig
	if req.ToolChoice != nil && req.ToolChoice.Type == "tool" && req.ToolChoice.Name != "" {
		gReq.ToolConfig = &ToolConfig{AllowedFunctionNames: []string{req.ToolChoice.Name}}
	}
	return gReq, nil
}

// performGeminiRequest executes HTTP call to Gemini endpoint.
func performGeminiRequest(model string, gReq *GenerationRequest, stream bool) (*http.Response, error) {
	client := &http.Client{Timeout: 0} // no timeout for stream; rely on context cancel

	endpoint := "/models/" + model
	if stream {
		endpoint += ":streamGenerateContent"
	} else {
		endpoint += ":generateContent"
	}
	url := fmt.Sprintf("%s%s", geminiBaseURL, endpoint)

	// Marshal request
	body, err := json.Marshal(gReq)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", geminiAPIKey)

	return client.Do(req)
}

// waitIfBackoff blocks until the global backoff window expires.
func waitIfBackoff(ctx context.Context) error {
	for {
		backoffMutex.Lock()
		until := backoffUntil
		backoffMutex.Unlock()
		if until.IsZero() || time.Now().After(until) {
			return nil
		}
		select {
		case <-time.After(time.Until(until)):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// setBackoff sets/extends the backoff window.
func setBackoff(d time.Duration) {
	backoffMutex.Lock()
	expiry := time.Now().Add(d)
	if backoffUntil.IsZero() || expiry.After(backoffUntil) {
		backoffUntil = expiry
	}
	backoffMutex.Unlock()
}

// performGeminiRequestWithRetry wraps performGeminiRequest with retry/backoff.
func performGeminiRequestWithRetry(model string, gReq *GenerationRequest, stream bool) (*http.Response, error) {
	attemptInternal := 0
	attemptRate := 0

	for {
		// honour global backoff
		if err := waitIfBackoff(context.Background()); err != nil {
			return nil, err
		}

		resp, err := performGeminiRequest(model, gReq, stream)
		if err != nil {
			// network-level error treat as internal server error
			if attemptInternal >= internalErrorMaxRetries {
				return nil, err
			}
			delay := internalErrorInitialDelay * time.Duration(1<<attemptInternal)
			attemptInternal++
			setBackoff(delay)
			log.Printf("[WARN] network error, retrying in %.0fs (attempt %d)", delay.Seconds(), attemptInternal)
			time.Sleep(delay)
			continue
		}

		// Success codes
		if resp.StatusCode < 400 {
			return resp, nil
		}

		// read up to 1KB of body for logging then close
		b := make([]byte, 1024)
		n, _ := resp.Body.Read(b)
		resp.Body.Close()
		log.Printf("[WARN] Upstream %d: %s", resp.StatusCode, strings.TrimSpace(string(b[:n])))

		switch resp.StatusCode {
		case 429:
			if attemptRate >= rateLimitMaxRetries {
				return resp, fmt.Errorf("rate limit reached (exceeded retries)")
			}
			attemptRate++
			delay := rateLimitDefaultDelay
			setBackoff(delay)
			log.Printf("[INFO] rate limited, retrying in %.0fs (attempt %d)", delay.Seconds(), attemptRate)
			time.Sleep(delay)
			continue
		default:
			if resp.StatusCode >= 500 {
				if attemptInternal >= internalErrorMaxRetries {
					return resp, fmt.Errorf("internal error %d after retries", resp.StatusCode)
				}
				delay := internalErrorInitialDelay * time.Duration(1<<attemptInternal)
				attemptInternal++
				setBackoff(delay)
				time.Sleep(delay)
				continue
			}
			// non-retriable
			return resp, fmt.Errorf("upstream error status %d", resp.StatusCode)
		}
	}
}

// ----- Handlers -----

func handleRoot(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, map[string]string{"message": "Anthropic-Compatible Proxy for Google Gemini (Go Edition)"})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/sysmon/health/"), "/")
	if len(parts) > 0 {
		t := parts[0]
		if t == "liveness" || t == "readiness" {
			respondJSON(w, map[string]string{"status": "UP"})
			return
		}
	}
	http.NotFound(w, r)
}

func handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req MessagesRequest

	// Optionally log request headers
	if logMessagesHeaders {
		hdr, _ := json.Marshal(r.Header)
		log.Printf("[REQ_HEADERS] /v1/messages: %s", string(hdr))
	}

	if logMessagesBody {
		// Read body for logging and decoding
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		log.Printf("[REQ_BODY] /v1/messages: %s", string(bodyBytes))
		if err := json.Unmarshal(bodyBytes, &req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
	} else {
		// Decode directly without extra copy
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
	}

	// Allocate unique request ID for log correlation
	reqID := nextRequestID()
	start := time.Now()
	geminiModel := mapRequestedModel(req.Model)
	logRequest(reqID, r.URL.Path, geminiModel, len(req.Tools), len(req.Messages))

	// ---- Session-based Langfuse trace management ----
	session, isNewSession := sessionManager.getOrCreateSession(r, geminiModel, len(req.Tools), len(req.Messages))

	// Extract first user message for session context
	var userQuery string
	if len(req.Messages) > 0 {
		if content, ok := req.Messages[0].Content.(string); ok {
			userQuery = content
			if len(userQuery) > 100 {
				userQuery = userQuery[:97] + "..."
			}
		}
	}

	// Create span for this specific request within the session
	spanName := fmt.Sprintf("request_%d", session.RequestCount)
	if userQuery != "" && isNewSession {
		spanName = fmt.Sprintf("query: %s", userQuery)
	}

	// Prepare span input (user request)
	spanInput := map[string]interface{}{
		"model":      req.Model,
		"max_tokens": req.MaxTokens,
		"messages":   req.Messages,
	}
	if len(req.Tools) > 0 {
		spanInput["tools"] = len(req.Tools)
	}

	spanID := lfStartSpan(session.TraceID, spanName, spanInput)

	log.Printf("[SESSION] Req[%d] in session %s (span: %s, new: %v)",
		reqID, session.TraceID[:8], spanID[:8], isNewSession)

	gReq, err := convertToGemini(&req, geminiModel)
	if err != nil {
		lfFinishSpan(spanID, map[string]interface{}{"error": err.Error()}, map[string]interface{}{"error": err.Error()})
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	upStart := time.Now()
	generationID := lfStartGeneration(session.TraceID, "gemini_generate", geminiModel, gReq, spanID)

	resp, err := performGeminiRequestWithRetry(geminiModel, gReq, req.Stream)
	if err != nil {
		lfFinishGeneration(generationID, nil, nil, map[string]interface{}{"error": err.Error()})
		lfFinishSpan(spanID, map[string]interface{}{"error": err.Error()}, map[string]interface{}{"error": err.Error()})
		http.Error(w, fmt.Sprintf("upstream error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Forward stream if requested
	if req.Stream {
		// SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// For streaming, we just handle normally and note it in output
		handleGeminiStreamToAnthropic(reqID, w, resp.Body)

		// Finish generation after stream completes
		latencyMs := time.Since(upStart).Milliseconds()

		// Prepare generation output for streaming
		generationOutput := map[string]interface{}{
			"stream": true,
			"status": resp.StatusCode,
			"note":   "Stream content processed in real-time",
		}

		lfFinishGeneration(generationID, generationOutput, nil, map[string]interface{}{
			"status":     resp.StatusCode,
			"latency_ms": latencyMs,
			"model":      geminiModel,
			"stream":     true,
		})

		log.Printf("[INFO][%d] Stream completed in %.2fs", reqID, time.Since(start).Seconds())
		lfFinishSpan(spanID, "stream completed", map[string]interface{}{
			"stream_completed": true,
			"duration_ms":      time.Since(start).Milliseconds(),
		})
		return
	}

	// Finish generation for non-streaming requests
	latencyMs := time.Since(upStart).Milliseconds()

	// Non-streaming – parse upstream JSON and translate to Anthropic blocks
	// Read the full response body for logging
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		lfFinishSpan(spanID, map[string]interface{}{"error": fmt.Sprintf("read body: %v", err)}, map[string]interface{}{"error": fmt.Sprintf("read body: %v", err)})
		http.Error(w, "failed to read upstream response", http.StatusBadGateway)
		return
	}

	// Parse the upstream response
	var upstream struct {
		Candidates []struct {
			Content Content `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(responseBody, &upstream); err != nil {
		lfFinishSpan(spanID, map[string]interface{}{"error": fmt.Sprintf("decode upstream: %v", err)}, map[string]interface{}{"error": fmt.Sprintf("decode upstream: %v", err)})
		http.Error(w, "failed to decode upstream", http.StatusBadGateway)
		return
	}

	// Store the raw upstream response for generation output
	var rawUpstreamResponse interface{}
	json.Unmarshal(responseBody, &rawUpstreamResponse)

	var blocks []interface{}
	if len(upstream.Candidates) > 0 {
		for idx, p := range upstream.Candidates[0].Content.Parts {
			if p.Text != "" {
				blocks = append(blocks, map[string]interface{}{
					"type": "text",
					"text": p.Text,
				})
			} else if p.FunctionCall != nil {
				log.Printf("[TOOL][%d] %s %s", reqID, p.FunctionCall.Name, summarizeArgs(p.FunctionCall.Name, p.FunctionCall.Args))
				// parse args JSON into map[string]interface{}
				var argsMap map[string]interface{}
				if err := json.Unmarshal(p.FunctionCall.Args, &argsMap); err != nil {
					argsMap = map[string]interface{}{"_raw": string(p.FunctionCall.Args)}
				}
				blocks = append(blocks, map[string]interface{}{
					"type":  "tool_use",
					"id":    fmt.Sprintf("tool_%d", idx),
					"name":  p.FunctionCall.Name,
					"input": argsMap,
				})
			}
		}
	}

	result := MessagesResponse{
		ID:      fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Model:   req.Model,
		Role:    "assistant",
		Content: blocks,
	}

	// Update generation with output - include raw response if enabled
	generationOutput := map[string]interface{}{
		"anthropic_format": result.Content,
		"processed_blocks": len(blocks),
	}

	// Optionally include full raw Gemini response for debugging
	if logFullOutput {
		generationOutput["raw_gemini_response"] = rawUpstreamResponse
	}

	lfFinishGeneration(generationID, generationOutput, nil, map[string]interface{}{
		"status":         resp.StatusCode,
		"latency_ms":     latencyMs,
		"model":          geminiModel,
		"response_id":    result.ID,
		"content_blocks": len(blocks),
	})

	respondJSON(w, result)
	log.Printf("[INFO][%d] Completed in %.2fs", reqID, time.Since(start).Seconds())

	// Finish span with request completion info
	spanMetadata := map[string]interface{}{
		"request_completed": true,
		"duration_ms":       time.Since(start).Milliseconds(),
		"content_blocks":    len(blocks),
		"model":             geminiModel,
	}

	// Add first text output for context
	if len(blocks) > 0 {
		if firstBlock, ok := blocks[0].(map[string]interface{}); ok {
			if text, exists := firstBlock["text"]; exists {
				outputText := fmt.Sprintf("%v", text)
				if len(outputText) > 200 {
					outputText = outputText[:197] + "..."
				}
				spanMetadata["output_preview"] = outputText
			}
		}
	}

	lfFinishSpan(spanID, result, spanMetadata)
}

func handleCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req TokenCountRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// Allocate unique request ID for log correlation
	reqID := nextRequestID()
	start := time.Now()
	geminiModel := mapRequestedModel(req.Model)

	logRequest(reqID, r.URL.Path, geminiModel, 0, len(req.Messages))

	// Prepare upstream request
	contents := make([]Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		if text, ok := m.Content.(string); ok {
			contents = append(contents, Content{Role: m.Role, Parts: []Part{{Text: text}}})
		}
	}

	upstreamReq := TokenCountRequestUpstream{Contents: contents}
	body, _ := json.Marshal(upstreamReq)

	url := fmt.Sprintf("%s/models/%s:countTokens", geminiBaseURL, geminiModel)
	httpReq, err := http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", geminiAPIKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	var upstreamResp TokenCountResponseUpstream
	if err := json.NewDecoder(resp.Body).Decode(&upstreamResp); err != nil {
		http.Error(w, "failed to decode upstream", http.StatusBadGateway)
		return
	}

	respondJSON(w, TokenCountResponse{InputTokens: upstreamResp.TotalTokens})
	log.Printf("[INFO][%d] Token count completed in %.2fs", reqID, time.Since(start).Seconds())
}

// respondJSON writes JSON with proper headers.
func respondJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	if err := enc.Encode(v); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/", handleRoot)
	mux.HandleFunc("/sysmon/health/", handleHealth)
	mux.HandleFunc("/v1/messages", handleMessages)
	mux.HandleFunc("/v1/messages/count_tokens", handleCountTokens)

	addr := getEnv("PORT", "8082")
	if geminiBaseURL != defaultGeminiBaseURL {
		log.Printf("[INFO] GEMINI_BASE_URL=%s", geminiBaseURL)
	}
	log.Printf("[INFO] Starting server on :%s (Big=%s Small=%s)\n", addr, defaultBigModel, defaultSmallModel)
	if err := http.ListenAndServe(":"+addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// ----- Tool / Function declarations -----

// FunctionDeclaration mirrors Gemini JSON schema

type FunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type Tool struct {
	// In Gemini, each tool contains array field functionDeclarations
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations"`
}

// ToolChoice config (simplified)

type ToolChoice struct {
	Type string `json:"type"`           // auto | any | tool
	Name string `json:"name,omitempty"` // when Type==tool
}

// ToolDescriptor represents incoming Anthropic tool definition
type ToolDescriptor struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// ToolConfig restricts allowed functions
type ToolConfig struct {
	AllowedFunctionNames []string `json:"allowedFunctionNames"`
}

// cleanGeminiSchema removes unsupported fields
func cleanGeminiSchema(m map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{})
	for k, v := range m {
		// Drop unsupported keys
		if k == "additionalProperties" || k == "default" || strings.HasPrefix(k, "$") {
			continue
		}

		switch val := v.(type) {
		case map[string]interface{}:
			out[k] = cleanGeminiSchema(val)
		case []interface{}:
			var arr []interface{}
			for _, item := range val {
				if sub, ok := item.(map[string]interface{}); ok {
					arr = append(arr, cleanGeminiSchema(sub))
				} else {
					arr = append(arr, item)
				}
			}
			out[k] = arr
		default:
			out[k] = v
		}
	}

	// Remove unsupported format from string types
	if t, ok := out["type"].(string); ok && t == "string" {
		if f, okFmt := out["format"].(string); okFmt {
			if f != "enum" && f != "date-time" {
				delete(out, "format")
			}
		}
	}
	return out
}

// writeSSE writes a Server-Sent Event with optional event name.
func writeSSE(w http.ResponseWriter, event string, data interface{}) {
	if event != "" {
		fmt.Fprintf(w, "event: %s\n", event)
	}
	b, _ := json.Marshal(data)
	fmt.Fprintf(w, "data: %s\n\n", string(b))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// handleGeminiStreamToAnthropic converts Gemini streaming JSON chunks to Anthropic SSE format.
func handleGeminiStreamToAnthropic(reqID uint64, w http.ResponseWriter, body io.ReadCloser) {
	defer body.Close()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 1024*64), 1024*1024)

	msgID := fmt.Sprintf("msg_%d", time.Now().UnixNano())

	// Emit initial events
	writeSSE(w, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      msgID,
			"role":    "assistant",
			"content": []interface{}{},
		},
	})
	writeSSE(w, "content_block_start", map[string]interface{}{
		"type":         "content_block_start",
		"index":        0,
		"content_type": "text",
	})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var chunk struct {
			Candidates []struct {
				Content      Content `json:"content"`
				FinishReason string  `json:"finishReason"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue // skip malformed
		}
		if len(chunk.Candidates) == 0 || len(chunk.Candidates[0].Content.Parts) == 0 {
			continue
		}
		part := chunk.Candidates[0].Content.Parts[0]
		if part.Text != "" {
			writeSSE(w, "content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]string{"text": part.Text},
			})
		} else if part.FunctionCall != nil {
			// Log tool invocation
			log.Printf("[TOOL][%d] %s %s", reqID, part.FunctionCall.Name, summarizeArgs(part.FunctionCall.Name, part.FunctionCall.Args))
			// Forward as tool_use block delta if desired (not mandatory for logging)
			writeSSE(w, "content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]interface{}{ // minimal
					"tool_call": part.FunctionCall.Name,
				},
			})
		}

		// Check for finish
		if fin := chunk.Candidates[0].FinishReason; fin != "" {
			stopReason := strings.ToLower(fin)
			if stopReason == "stop" {
				stopReason = "end_turn"
			}
			writeSSE(w, "content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": 0,
			})
			writeSSE(w, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason": stopReason,
				},
			})
			writeSSE(w, "message_stop", map[string]interface{}{"type": "message_stop"})
			return
		}
	}

	// stream ended without explicit finish
	writeSSE(w, "content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": 0})
	writeSSE(w, "message_stop", map[string]interface{}{"type": "message_stop"})
}

// logRequest prints path, model used, tools/messages counts.
func logRequest(id uint64, path, model string, tools, messages int) {
	log.Printf("[REQ][%d] %s -> %s (%d tools, %d messages)", id, path, model, tools, messages)
}

func mapRole(r string) string {
	if r == "user" {
		return "user"
	}
	return "model"
}

// nextRequestID atomically increments and returns the next request identifier.
func nextRequestID() uint64 {
	return atomic.AddUint64(&reqIDCounter, 1)
}

// summarizeArgs produces a concise string representation of tool arguments for logging.
func summarizeArgs(name string, raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var generic interface{}
	if err := json.Unmarshal(raw, &generic); err != nil {
		return ""
	}

	// Special-case bash tool – show full command.
	m, isMap := generic.(map[string]interface{})
	if isMap {
		lname := strings.ToLower(name)
		if lname == "bash" || strings.Contains(lname, "bash") {
			if cmd, ok := m["command"]; ok {
				return fmt.Sprintf("\"%v\"", cmd)
			}
		}
	}

	// Fallback: marshal compact and truncate if long.
	b, _ := json.Marshal(generic)
	s := string(b)
	if len(s) > 120 {
		s = s[:117] + "..."
	}
	return s
}
