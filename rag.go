package main

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Embedder превращает текст в вектор (шаги 3 и 5).
// Kind нужен, потому что у Яндекса разные модели для документов и для запросов,
// но обе кладут точки на одну и ту же карту.
type Kind int

const (
	KindDoc Kind = iota
	KindQuery
)

type Embedder interface {
	Embed(ctx context.Context, text string, kind Kind) ([]float64, error)
}

// Card — одна карточка библиотеки (шаги 1–2).
type Card struct {
	ID   int
	Text string
}

// entry — карточка вместе с её координатами (шаг 4).
type entry struct {
	Card
	Vec []float64
}

// Index — наше «хранилище». Для прототипа хватает слайса в памяти.
type Index struct {
	emb     Embedder
	entries []entry
}

func NewIndex(emb Embedder) *Index { return &Index{emb: emb} }

// Add — шаги 3–4: посчитать эмбеддинг карточки и положить в хранилище.
// Делается один раз, заранее.
func (ix *Index) Add(ctx context.Context, cards ...Card) error {
	for _, c := range cards {
		v, err := ix.emb.Embed(ctx, c.Text, KindDoc)
		if err != nil {
			return fmt.Errorf("embed card %d: %w", c.ID, err)
		}
		ix.entries = append(ix.entries, entry{Card: c, Vec: v})
	}
	return nil
}

// Hit — найденная карточка и её близость к вопросу.
type Hit struct {
	Card
	Score float64
}

// Search — шаги 5–6: вопрос → вектор → косинус с каждой карточкой → топ-K.
// minScore — порог: всё, что ниже, считаем «не по теме» и выкидываем.
func (ix *Index) Search(ctx context.Context, question string, topK int, minScore float64) ([]Hit, error) {
	q, err := ix.emb.Embed(ctx, question, KindQuery)
	if err != nil {
		return nil, fmt.Errorf("embed question: %w", err)
	}
	hits := make([]Hit, 0, len(ix.entries))
	for _, e := range ix.entries {
		hits = append(hits, Hit{Card: e.Card, Score: Cosine(q, e.Vec)})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if len(hits) > topK {
		hits = hits[:topK]
	}
	// Порог отсекаем после сортировки: может остаться 0 карточек — и это нормально.
	out := hits[:0]
	for _, h := range hits {
		if h.Score >= minScore {
			out = append(out, h)
		}
	}
	return out, nil
}

// Cosine — косинус угла между векторами: (a·b) / (|a|·|b|).
func Cosine(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// BuildPrompt — шаг 7: инструкция + найденные карточки с номерами + вопрос.
func BuildPrompt(question string, hits []Hit) string {
	var b strings.Builder
	b.WriteString("Ты аналитик отзывов. Отвечай ТОЛЬКО по отзывам ниже.\n")
	b.WriteString("Если ответа в них нет, напиши \"нет данных\".\n")
	b.WriteString("Указывай номера отзывов в квадратных скобках.\n\nОтзывы:\n")
	for _, h := range hits {
		fmt.Fprintf(&b, "[%d] %s\n", h.ID, h.Text)
	}
	fmt.Fprintf(&b, "\nВопрос: %s\n", question)
	return b.String()
}
