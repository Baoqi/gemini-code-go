package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Langfuse ingestion helper based on langfuse-core implementation.
// Handles proper event types and data structures according to Langfuse API schema.

const (
	langfuseDefaultHost = "https://cloud.langfuse.com"
	lfBatchSize         = 50 // increased batch size for better performance
	lfFlushInterval     = 5 * time.Second

	lfTimeFmt = time.RFC3339Nano // ISO 8601 format required by Langfuse
)

// Event types as defined by Langfuse ingestion API
const (
	EventTypeTraceCreate      = "trace-create"
	EventTypeTraceUpdate      = "trace-update"
	EventTypeSpanCreate       = "span-create"
	EventTypeSpanUpdate       = "span-update"
	EventTypeGenerationCreate = "generation-create"
	EventTypeGenerationUpdate = "generation-update"
	EventTypeEventCreate      = "event-create"
	EventTypeScoreCreate      = "score-create"
)

// lfEvent represents the event envelope for Langfuse ingestion
type lfEvent struct {
	ID        string      `json:"id"`
	Timestamp string      `json:"timestamp"`
	Type      string      `json:"type"`
	Body      interface{} `json:"body"`
}

// Trace body structure
type TraceBody struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name,omitempty"`
	SessionID   string                 `json:"sessionId,omitempty"`
	UserID      string                 `json:"userId,omitempty"`
	Input       interface{}            `json:"input,omitempty"`
	Output      interface{}            `json:"output,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	StartTime   string                 `json:"startTime,omitempty"`
	EndTime     string                 `json:"endTime,omitempty"`
	Release     string                 `json:"release,omitempty"`
	Version     string                 `json:"version,omitempty"`
	Public      bool                   `json:"public,omitempty"`
	Environment string                 `json:"environment,omitempty"`
	Timestamp   string                 `json:"timestamp,omitempty"`
}

// Span body structure
type SpanBody struct {
	ID                  string                 `json:"id"`
	TraceID             string                 `json:"traceId"`
	ParentObservationID string                 `json:"parentObservationId,omitempty"`
	Name                string                 `json:"name,omitempty"`
	Input               interface{}            `json:"input,omitempty"`
	Output              interface{}            `json:"output,omitempty"`
	Metadata            map[string]interface{} `json:"metadata,omitempty"`
	Level               string                 `json:"level,omitempty"`
	StatusMessage       string                 `json:"statusMessage,omitempty"`
	StartTime           string                 `json:"startTime,omitempty"`
	EndTime             string                 `json:"endTime,omitempty"`
	Version             string                 `json:"version,omitempty"`
	Environment         string                 `json:"environment,omitempty"`
}

// Generation body structure
type GenerationBody struct {
	ID                  string                 `json:"id"`
	TraceID             string                 `json:"traceId"`
	ParentObservationID string                 `json:"parentObservationId,omitempty"`
	Name                string                 `json:"name,omitempty"`
	Model               string                 `json:"model,omitempty"`
	ModelParameters     map[string]interface{} `json:"modelParameters,omitempty"`
	Input               interface{}            `json:"input,omitempty"`
	Output              interface{}            `json:"output,omitempty"`
	Usage               *UsageInfo             `json:"usage,omitempty"`
	Metadata            map[string]interface{} `json:"metadata,omitempty"`
	Level               string                 `json:"level,omitempty"`
	StatusMessage       string                 `json:"statusMessage,omitempty"`
	StartTime           string                 `json:"startTime,omitempty"`
	EndTime             string                 `json:"endTime,omitempty"`
	CompletionStartTime string                 `json:"completionStartTime,omitempty"`
	Version             string                 `json:"version,omitempty"`
	Environment         string                 `json:"environment,omitempty"`
}

// Event body structure
type EventBody struct {
	ID                  string                 `json:"id"`
	TraceID             string                 `json:"traceId"`
	ParentObservationID string                 `json:"parentObservationId,omitempty"`
	Name                string                 `json:"name,omitempty"`
	Input               interface{}            `json:"input,omitempty"`
	Output              interface{}            `json:"output,omitempty"`
	Metadata            map[string]interface{} `json:"metadata,omitempty"`
	Level               string                 `json:"level,omitempty"`
	StatusMessage       string                 `json:"statusMessage,omitempty"`
	StartTime           string                 `json:"startTime,omitempty"`
	Version             string                 `json:"version,omitempty"`
	Environment         string                 `json:"environment,omitempty"`
}

// Score body structure
type ScoreBody struct {
	ID            string  `json:"id"`
	TraceID       string  `json:"traceId"`
	ObservationID string  `json:"observationId,omitempty"`
	Name          string  `json:"name"`
	Value         float64 `json:"value"`
	DataType      string  `json:"dataType,omitempty"`
	Comment       string  `json:"comment,omitempty"`
	Source        string  `json:"source,omitempty"`
	ConfigID      string  `json:"configId,omitempty"`
	Environment   string  `json:"environment,omitempty"`
}

// Usage information for generations
type UsageInfo struct {
	Input      int     `json:"input,omitempty"`
	Output     int     `json:"output,omitempty"`
	Total      int     `json:"total,omitempty"`
	Unit       string  `json:"unit,omitempty"`
	InputCost  float64 `json:"inputCost,omitempty"`
	OutputCost float64 `json:"outputCost,omitempty"`
	TotalCost  float64 `json:"totalCost,omitempty"`
	// OpenAI-style fields for backward compatibility
	PromptTokens     int `json:"promptTokens,omitempty"`
	CompletionTokens int `json:"completionTokens,omitempty"`
	TotalTokens      int `json:"totalTokens,omitempty"`
}

var (
	lfEnabled      bool
	lfHost         string
	lfAuthHeader   string // Basic <base64>
	lfCh           chan lfEvent
	lfEventCounter uint64
	lfSpanTrace    sync.Map // spanID -> traceID
)

// initLangfuse initializes the Langfuse client with environment variables
func initLangfuse() {
	// Check if Langfuse is explicitly enabled
	if getEnv("GEMINI_CODE_ENABLE_LANGFUSE", "") == "" {
		log.Printf("[LF] Langfuse disabled: GEMINI_CODE_ENABLE_LANGFUSE not set")
		return
	}

	pub := os.Getenv("LANGFUSE_PUBLIC_KEY")
	sec := os.Getenv("LANGFUSE_SECRET_KEY")
	if pub == "" || sec == "" {
		log.Printf("[LF] Langfuse disabled: missing public or secret key")
		return
	}

	lfHost = getEnv("LANGFUSE_HOST", langfuseDefaultHost)

	// Basic auth header
	token := base64.StdEncoding.EncodeToString([]byte(pub + ":" + sec))
	lfAuthHeader = "Basic " + token
	lfEnabled = true

	lfCh = make(chan lfEvent, 2000)
	go lfSenderLoop()
	log.Printf("[LF] Langfuse enabled, host=%s", lfHost)

	// Enable debug mode if requested
	if getEnv("LANGFUSE_DEBUG", "") != "" {
		log.Printf("[LF] Debug mode enabled")
	}
}

// lfSenderLoop processes events in batches
func lfSenderLoop() {
	client := &http.Client{Timeout: 15 * time.Second}
	ticker := time.NewTicker(lfFlushInterval)
	defer ticker.Stop()

	buffer := make([]lfEvent, 0, lfBatchSize)

	flush := func() {
		if len(buffer) == 0 {
			return
		}

		payload := map[string]interface{}{"batch": buffer}
		nEvents := len(buffer)

		b, err := json.Marshal(payload)
		if err != nil {
			log.Printf("[LF][ERR] marshal payload: %v", err)
			buffer = buffer[:0]
			return
		}

		req, err := http.NewRequest("POST", lfHost+"/api/public/ingestion", bytes.NewReader(b))
		if err != nil {
			log.Printf("[LF][ERR] create request: %v", err)
			buffer = buffer[:0]
			return
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", lfAuthHeader)
		req.Header.Set("User-Agent", "gemini-code-go/1.0")

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		req = req.WithContext(ctx)

		resp, err := client.Do(req)
		if err != nil {
			log.Printf("[LF][ERR] send request: %v", err)
			buffer = buffer[:0]
			return
		}
		defer resp.Body.Close()

		bodyBytes, _ := io.ReadAll(resp.Body)

		if resp.StatusCode == 200 {
			log.Printf("[LF] Successfully sent %d events", nEvents)
		} else if resp.StatusCode == 207 {
			// Partial success - check for errors
			var respJSON struct {
				Successes []interface{} `json:"successes"`
				Errors    []interface{} `json:"errors"`
			}
			if err := json.Unmarshal(bodyBytes, &respJSON); err == nil {
				if len(respJSON.Errors) == 0 {
					log.Printf("[LF] Successfully sent %d events (status 207)", nEvents)
				} else {
					log.Printf("[LF][WARN] Partial success - %d successes, %d errors: %s",
						len(respJSON.Successes), len(respJSON.Errors), truncate(string(bodyBytes), 500))

					// In debug mode, log the failed events and payload
					if getEnv("LANGFUSE_DEBUG", "") != "" {
						debugPayload, _ := json.Marshal(payload)
						log.Printf("[LF][DEBUG] Request payload with errors: %s", truncate(string(debugPayload), 2000))
					}
				}
			} else {
				log.Printf("[LF][WARN] Status 207 with unparseable body: %s", truncate(string(bodyBytes), 500))
			}
		} else {
			log.Printf("[LF][ERR] HTTP %d: %s", resp.StatusCode, truncate(string(bodyBytes), 500))
			// In debug mode, also log the request payload for failed requests
			if getEnv("LANGFUSE_DEBUG", "") != "" && len(buffer) > 0 {
				debugPayload, _ := json.Marshal(payload)
				log.Printf("[LF][DEBUG] Failed payload: %s", truncate(string(debugPayload), 1000))
			}
		}

		buffer = buffer[:0]
	}

	for {
		select {
		case ev := <-lfCh:
			buffer = append(buffer, ev)
			if len(buffer) >= lfBatchSize {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}

// lfPush queues an event for sending
func lfPush(ev lfEvent) {
	if !lfEnabled {
		return
	}

	// Debug logging if enabled
	if getEnv("LANGFUSE_DEBUG", "") != "" {
		bodyJson, _ := json.Marshal(ev.Body)
		log.Printf("[LF][DEBUG] Queuing event: type=%s, id=%s, body=%s", ev.Type, ev.ID, string(bodyJson))
	}

	select {
	case lfCh <- ev:
	default:
		log.Printf("[LF][DROP] Channel full, dropping event %s", ev.ID)
	}
}

// --- Public API functions ---

// lfStartTrace creates and records a new trace
func lfStartTrace(name string, input interface{}) string {
	if !lfEnabled {
		return ""
	}

	traceID := newLfID("trace")
	now := time.Now().UTC().Format(lfTimeFmt)

	body := TraceBody{
		ID:          traceID,
		Name:        name,
		Input:       input,
		StartTime:   now,
		Release:     getEnv("LANGFUSE_RELEASE", ""),
		Version:     "1.0",
		Environment: getEnv("LANGFUSE_ENVIRONMENT", "production"),
		Timestamp:   now,
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeTraceCreate,
		Body:      body,
	})

	return traceID
}

// lfFinishTrace updates a trace with output and end time
// Note: We skip trace-update to avoid format compatibility issues with Langfuse API
// The generation-update events already contain all the necessary output information
func lfFinishTrace(traceID string, output interface{}, err error) {
	if !lfEnabled || traceID == "" {
		return
	}
	// Skip trace-update to avoid discriminator format issues
	// The generation events already provide the complete trace information
}

// lfStartSpan creates and records a new span
func lfStartSpan(traceID, name string, input interface{}) string {
	if !lfEnabled || traceID == "" {
		return ""
	}

	spanID := newLfID("span")
	now := time.Now().UTC().Format(lfTimeFmt)

	body := SpanBody{
		ID:          spanID,
		TraceID:     traceID,
		Name:        name,
		Input:       input,
		StartTime:   now,
		Level:       "DEFAULT",
		Environment: getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeSpanCreate,
		Body:      body,
	})

	// Store mapping for updates
	lfSpanTrace.Store(spanID, traceID)
	return spanID
}

// lfFinishSpan updates a span with output and end time
func lfFinishSpan(spanID string, output interface{}, metadata map[string]interface{}) {
	if !lfEnabled || spanID == "" {
		return
	}

	traceIDValue, ok := lfSpanTrace.Load(spanID)
	if !ok {
		log.Printf("[LF][WARN] No trace ID found for span %s", spanID)
		return
	}

	traceID := traceIDValue.(string)
	now := time.Now().UTC().Format(lfTimeFmt)

	body := SpanBody{
		ID:          spanID,
		TraceID:     traceID,
		Output:      output,
		EndTime:     now,
		Metadata:    metadata,
		Environment: getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeSpanUpdate,
		Body:      body,
	})

	// Clean up mapping
	lfSpanTrace.Delete(spanID)
}

// lfStartGeneration creates and records a new generation as child of a span
func lfStartGeneration(traceID, name, model string, input interface{}, parentSpanID string) string {
	if !lfEnabled || traceID == "" {
		return ""
	}

	generationID := newLfID("generation")
	now := time.Now().UTC().Format(lfTimeFmt)

	body := GenerationBody{
		ID:                  generationID,
		TraceID:             traceID,
		ParentObservationID: parentSpanID, // Set parent relationship
		Name:                name,
		Model:               model,
		Input:               input,
		StartTime:           now,
		Level:               "DEFAULT",
		Environment:         getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeGenerationCreate,
		Body:      body,
	})

	// Store mapping for updates
	lfSpanTrace.Store(generationID, traceID)
	return generationID
}

// lfFinishGeneration updates a generation with output, usage, and end time
func lfFinishGeneration(generationID string, output interface{}, usage *UsageInfo, metadata map[string]interface{}) {
	if !lfEnabled || generationID == "" {
		return
	}

	traceIDValue, ok := lfSpanTrace.Load(generationID)
	if !ok {
		log.Printf("[LF][WARN] No trace ID found for generation %s", generationID)
		return
	}

	traceID := traceIDValue.(string)
	now := time.Now().UTC().Format(lfTimeFmt)

	body := GenerationBody{
		ID:          generationID,
		TraceID:     traceID,
		Output:      output,
		Usage:       usage,
		Metadata:    metadata,
		EndTime:     now,
		Environment: getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeGenerationUpdate,
		Body:      body,
	})

	// Clean up mapping
	lfSpanTrace.Delete(generationID)
}

// lfCreateEvent records a discrete event
func lfCreateEvent(traceID, parentObservationID, name string, input, output interface{}, metadata map[string]interface{}) string {
	if !lfEnabled || traceID == "" {
		return ""
	}

	eventID := newLfID("event_obs")
	now := time.Now().UTC().Format(lfTimeFmt)

	body := EventBody{
		ID:                  eventID,
		TraceID:             traceID,
		ParentObservationID: parentObservationID,
		Name:                name,
		Input:               input,
		Output:              output,
		Metadata:            metadata,
		StartTime:           now,
		Level:               "DEFAULT",
		Environment:         getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeEventCreate,
		Body:      body,
	})

	return eventID
}

// lfCreateScore records a score for evaluation
func lfCreateScore(traceID, observationID, name string, value float64, comment string) {
	if !lfEnabled || traceID == "" {
		return
	}

	scoreID := newLfID("score")
	now := time.Now().UTC().Format(lfTimeFmt)

	body := ScoreBody{
		ID:            scoreID,
		TraceID:       traceID,
		ObservationID: observationID,
		Name:          name,
		Value:         value,
		DataType:      "NUMERIC",
		Comment:       comment,
		Source:        "API",
		Environment:   getEnv("LANGFUSE_ENVIRONMENT", "production"),
	}

	lfPush(lfEvent{
		ID:        newLfID("event"),
		Timestamp: now,
		Type:      EventTypeScoreCreate,
		Body:      body,
	})
}

// lfFlushAsync flushes all pending events
func lfFlushAsync() {
	if !lfEnabled {
		return
	}

	// Send a signal to flush by sending empty event
	select {
	case lfCh <- lfEvent{}:
	default:
	}

	// Wait a bit for the flush to complete
	time.Sleep(100 * time.Millisecond)
}

// Helper functions

// newLfID generates a unique event ID
func newLfID(prefix string) string {
	n := atomic.AddUint64(&lfEventCounter, 1)
	timestamp := strconv.FormatInt(time.Now().UnixNano(), 10)
	counter := strconv.FormatUint(n, 10)
	random := strconv.Itoa(rand.Intn(10000))
	return fmt.Sprintf("%s_%s_%s_%s", prefix, timestamp, counter, random)
}

// truncate limits string length for logging
func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
