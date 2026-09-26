package unit_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/zerothorder"
)

func TestResidentTemplateFailurePreventsGeneration(t *testing.T) {
	operations := []struct {
		name string
		run  func(*services.LlamaEngine) error
	}{
		{"completion", func(engine *services.LlamaEngine) error {
			_, err := engine.Complete(context.Background(), "user content", 32)
			return err
		}},
		{"choice", func(engine *services.LlamaEngine) error {
			_, err := engine.ChoiceMargin(context.Background(), "choose", "A", []string{"A", "B"})
			return err
		}},
	}
	for _, response := range []struct {
		name, body string
		status     int
	}{
		{"unavailable", `{"error":"template unavailable"}`, http.StatusNotFound},
		{"empty-body", "", http.StatusOK},
		{"missing-prompt", `{}`, http.StatusOK},
		{"empty-prompt", `{"prompt":""}`, http.StatusOK},
		{"blank-prompt", `{"prompt":"  \n"}`, http.StatusOK},
		{"null-prompt", `{"prompt":null}`, http.StatusOK},
		{"wrong-type", `{"prompt":[1,2]}`, http.StatusOK},
		{"malformed", `{"prompt":`, http.StatusOK},
	} {
		for _, operation := range operations {
			t.Run(response.name+"/"+operation.name, func(t *testing.T) {
				calls := 0
				engine := services.NewAuthenticatedLlamaEngine("https://resident.invalid", "fixture-key")
				engine.Client.Transport = residentTemplateRoundTripper(func(r *http.Request) (*http.Response, error) {
					calls++
					if r.URL.Path != "/apply-template" || r.Header.Get("Authorization") != "Bearer fixture-key" {
						t.Fatal("template failure triggered generation or lost authentication")
					}
					return &http.Response{StatusCode: response.status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(response.body))}, nil
				})
				if err := operation.run(engine); err == nil {
					t.Fatal("invalid backend template was accepted")
				}
				if calls != 1 {
					t.Fatalf("template calls = %d; no retry or fallback is admitted", calls)
				}
			})
		}
	}
}

type residentTemplateRoundTripper func(*http.Request) (*http.Response, error)

func (f residentTemplateRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
