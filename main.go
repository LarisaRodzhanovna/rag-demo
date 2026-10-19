package main

import (
	"context"
	"fmt"
	"os"
)

// ToyEmbedder — игрушечная карта из разбора: 2 оси, «про доставку» и «про еду».
// Нужен, чтобы прогнать весь RAG без сети и денег и увидеть те же числа, что на картинке.
type ToyEmbedder map[string][]float64

func (t ToyEmbedder) Embed(_ context.Context, text string, _ Kind) ([]float64, error) {
	v, ok := t[text]
	if !ok {
		return nil, fmt.Errorf("toy embedder: unknown text %q", text)
	}
	return v, nil
}

func main() {
	ctx := context.Background()

	cards := []Card{
		{1, "Курьер опоздал на два часа"},
		{2, "Доставка задержалась, пицца остыла"},
		{3, "Пицца очень вкусная, тесто тонкое"},
		{4, "Сыра пожалели, начинки мало"},
		{5, "Привезли быстро, курьер вежливый"},
	}

	var emb Embedder = ToyEmbedder{
		cards[0].Text: {0.90, 0.10},
		cards[1].Text: {0.80, 0.50},
		cards[2].Text: {0.10, 0.90},
		cards[3].Text: {0.10, 0.80},
		cards[4].Text: {0.85, 0.20},
		"Почему недовольны доставкой?": {0.95, 0.15},
		"Что говорят про начинку?":     {0.15, 0.95},
		"Как лечить простуду?":         {-0.70, 0.10}, // не по теме: смотрит «в другую сторону»
	}

	// Переключение на настоящий Яндекс: YANDEX_API_KEY + YANDEX_FOLDER_ID.
	if key, folder := os.Getenv("YANDEX_API_KEY"), os.Getenv("YANDEX_FOLDER_ID"); key != "" && folder != "" {
		emb = &YandexEmbedder{APIKey: key, FolderID: folder}
		fmt.Println("Эмбеддинги: YandexGPT (настоящие, 256 чисел)")
	} else {
		fmt.Println("Эмбеддинги: игрушечные, 2 оси")
	}

	ix := NewIndex(emb)
	if err := ix.Add(ctx, cards...); err != nil {
		fmt.Println("ошибка:", err)
		os.Exit(1)
	}

	const (
		topK     = 3   // сколько карточек максимум отдаём в промпт
		minScore = 0.4 // ниже этого косинуса карточка считается «не по теме»
	)
	questions := []string{"Почему недовольны доставкой?", "Что говорят про начинку?", "Как лечить простуду?"}
	for _, q := range questions {
		hits, err := ix.Search(ctx, q, topK, minScore)
		if err != nil {
			fmt.Println("ошибка:", err)
			continue
		}
		fmt.Printf("\n=== %s ===\n", q)
		if len(hits) == 0 {
			fmt.Printf("Ничего релевантного (все ниже порога %.1f) → отвечаем «нет данных», LLM не вызываем\n", minScore)
			continue
		}
		for _, h := range hits {
			fmt.Printf("  %.3f  [%d] %s\n", h.Score, h.ID, h.Text)
		}
		if q == questions[0] {
			fmt.Println("\n--- промпт для LLM ---")
			fmt.Print(BuildPrompt(q, hits))
		}
	}
}
