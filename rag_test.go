package main

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
)

func TestCosine(t *testing.T) {
	cases := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"одно направление", []float64{1, 2}, []float64{2, 4}, 1},
		{"перпендикулярные", []float64{1, 0}, []float64{0, 1}, 0},
		{"противоположные", []float64{1, 0}, []float64{-1, 0}, -1},
		{"разная длина", []float64{1, 0}, []float64{1}, 0},
		{"пустые", nil, nil, 0},
		{"нулевой вектор", []float64{0, 0}, []float64{1, 1}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Cosine(c.a, c.b); math.Abs(got-c.want) > 1e-9 {
				t.Fatalf("Cosine(%v, %v) = %v, ожидали %v", c.a, c.b, got, c.want)
			}
		})
	}
}

// testIndex — три карточки на двух осях («доставка» и «еда») и игрушечный эмбеддер без сети.
func testIndex(t *testing.T) *Index {
	t.Helper()
	cards := []Card{
		{1, "доставка опоздала"},
		{2, "доставка быстрая"},
		{3, "вкусная пицца"},
	}
	emb := ToyEmbedder{
		cards[0].Text:   {0.9, 0.1},
		cards[1].Text:   {0.8, 0.3},
		cards[2].Text:   {0.1, 0.9},
		"про доставку":  {1, 0.1},
		"про лекарства": {-1, 0},
	}
	ix := NewIndex(emb)
	if err := ix.Add(context.Background(), cards...); err != nil {
		t.Fatal(err)
	}
	return ix
}

func ids(hits []Hit) []int {
	out := make([]int, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func TestSearchRanksByScoreAndCutsTopK(t *testing.T) {
	hits, err := testIndex(t).Search(context.Background(), "про доставку", 2, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(hits); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("ожидали карточки [1 2] по убыванию близости, получили %v", got)
	}
	if hits[0].Score < hits[1].Score {
		t.Fatalf("результаты не отсортированы: %v < %v", hits[0].Score, hits[1].Score)
	}
}

func TestSearchAppliesThreshold(t *testing.T) {
	// topK=3 позволяет вернуть все карточки, но карточка про еду ниже порога 0.5.
	hits, err := testIndex(t).Search(context.Background(), "про доставку", 3, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.ID == 3 {
			t.Fatalf("карточка про еду не должна пройти порог: %+v", h)
		}
	}
	if len(hits) != 2 {
		t.Fatalf("ожидали 2 карточки выше порога, получили %v", ids(hits))
	}
}

func TestSearchOffTopicReturnsNothing(t *testing.T) {
	hits, err := testIndex(t).Search(context.Background(), "про лекарства", 3, 0.4)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("вопрос не по теме должен дать пустую выдачу, получили %v", ids(hits))
	}
}

func TestEmbedErrorsAreWrapped(t *testing.T) {
	ix := testIndex(t)
	if _, err := ix.Search(context.Background(), "неизвестный вопрос", 3, 0.4); err == nil ||
		!strings.Contains(err.Error(), "embed question") {
		t.Fatalf("ожидали ошибку с контекстом embed question, получили %v", err)
	}
	if err := ix.Add(context.Background(), Card{ID: 7, Text: "нет в словаре"}); err == nil ||
		!strings.Contains(err.Error(), "embed card 7") {
		t.Fatalf("ожидали ошибку с номером карточки, получили %v", err)
	}
}

func TestBuildPrompt(t *testing.T) {
	p := BuildPrompt("про доставку", []Hit{{Card: Card{ID: 1, Text: "доставка опоздала"}}})
	for _, want := range []string{"[1] доставка опоздала", "Вопрос: про доставку", "нет данных"} {
		if !strings.Contains(p, want) {
			t.Errorf("в промпте нет %q:\n%s", want, p)
		}
	}
}

// roundTripFunc подменяет сеть: Yandex-эмбеддер проверяем без реального API.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func reply(status int, statusText, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     statusText,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

func TestYandexEmbedderRequestAndParsing(t *testing.T) {
	cases := []struct {
		kind      Kind
		wantModel string
	}{
		{KindDoc, "emb://b1gfolder/text-search-doc/latest"},
		{KindQuery, "emb://b1gfolder/text-search-query/latest"},
	}
	for _, c := range cases {
		var gotAuth, gotFolder, gotModel string
		rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
			gotAuth, gotFolder = r.Header.Get("Authorization"), r.Header.Get("x-folder-id")
			var body struct {
				ModelURI string `json:"modelUri"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			gotModel = body.ModelURI
			// числа могут прийти и числами, и строками
			return reply(http.StatusOK, "200 OK", `{"embedding":[0.5,"-0.25",1]}`), nil
		})
		y := &YandexEmbedder{APIKey: "k", FolderID: "b1gfolder", HTTP: &http.Client{Transport: rt}}

		vec, err := y.Embed(context.Background(), "текст", c.kind)
		if err != nil {
			t.Fatal(err)
		}
		if len(vec) != 3 || vec[0] != 0.5 || vec[1] != -0.25 || vec[2] != 1 {
			t.Fatalf("неверный разбор вектора: %v", vec)
		}
		if gotAuth != "Api-Key k" || gotFolder != "b1gfolder" || gotModel != c.wantModel {
			t.Fatalf("неверный запрос: auth=%q folder=%q model=%q", gotAuth, gotFolder, gotModel)
		}
	}
}

func TestYandexEmbedderErrors(t *testing.T) {
	newClient := func(status int, statusText, body string) *YandexEmbedder {
		rt := roundTripFunc(func(*http.Request) (*http.Response, error) { return reply(status, statusText, body), nil })
		return &YandexEmbedder{APIKey: "k", FolderID: "f", HTTP: &http.Client{Transport: rt}}
	}
	if _, err := newClient(http.StatusForbidden, "403 Forbidden", "denied").Embed(context.Background(), "x", KindDoc); err == nil ||
		!strings.Contains(err.Error(), "403") {
		t.Fatalf("ожидали ошибку со статусом 403, получили %v", err)
	}
	if _, err := newClient(http.StatusOK, "200 OK", `{"embedding":["abc"]}`).Embed(context.Background(), "x", KindDoc); err == nil ||
		!strings.Contains(err.Error(), "embedding[0]") {
		t.Fatalf("ожидали ошибку разбора embedding[0], получили %v", err)
	}
}
