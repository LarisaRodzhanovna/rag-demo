package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// YandexEmbedder — настоящий эмбеддер через Yandex Foundation Models.
// Модели: text-search-doc (для карточек) и text-search-query (для вопросов), 256 чисел.
// Сверь эндпоинт и формат ответа с актуальной документацией Yandex AI Studio.
type YandexEmbedder struct {
	APIKey   string // ключ сервисного аккаунта (роль ai.languageModels.user)
	FolderID string
	HTTP     *http.Client
}

const yandexEmbedURL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

func (y *YandexEmbedder) Embed(ctx context.Context, text string, kind Kind) ([]float64, error) {
	model := "text-search-doc"
	if kind == KindQuery {
		model = "text-search-query"
	}
	body, _ := json.Marshal(map[string]string{
		"modelUri": fmt.Sprintf("emb://%s/%s/latest", y.FolderID, model),
		"text":     text,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, yandexEmbedURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Api-Key "+y.APIKey)
	req.Header.Set("x-folder-id", y.FolderID)

	client := y.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("yandex embeddings: %s: %s", resp.Status, raw)
	}

	// Числа могут прийти как числами, так и строками — разбираем оба варианта.
	var parsed struct {
		Embedding []json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode embedding: %w", err)
	}
	vec := make([]float64, len(parsed.Embedding))
	for i, r := range parsed.Embedding {
		f, err := strconv.ParseFloat(strings.Trim(string(r), `"`), 64)
		if err != nil {
			return nil, fmt.Errorf("embedding[%d]: %w", i, err)
		}
		vec[i] = f
	}
	return vec, nil
}
