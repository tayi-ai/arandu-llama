package teachers

import (
	"context"
	"errors"
	"math"
	"slices"
	"sync"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

type nativeProducer struct {
	mu       sync.Mutex
	factory  *NativeFactory
	job      *nativeJob
	model    *llama.TeacherModel
	cache    map[int][]fusioncache.Signal
	closed   bool
	closeErr error
}

var _ fusioncache.Producer = (*nativeProducer)(nil)

func (p *nativeProducer) Identity() fusioncache.ModelIdentity { return p.job.expected.Teacher }

func (p *nativeProducer) TeacherForce(ctx context.Context, request fusioncache.PrefixRequest) (fusioncache.Signal, error) {
	if ctx == nil {
		return fusioncache.Signal{}, rejected("context required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return fusioncache.Signal{}, errors.Join(rejected("producer closed"), p.closeErr)
	}
	if err := ctx.Err(); err != nil {
		return fusioncache.Signal{}, err
	}
	if request.TargetIndex < 1 || len(request.Tokens) > p.job.config.Capture.ContextTokens ||
		request.TargetIndex != len(request.Tokens) || !validDigest(request.PrefixSHA256) || request.PrefixSHA256 != fusioncache.PrefixDigest(request.Tokens) ||
		!slices.Equal(request.Features, p.job.expected.Features) {
		return fusioncache.Signal{}, rejected("request prefix, target index or features differ")
	}
	reference, exists := p.job.rows[request.PrefixSHA256]
	if !exists {
		return fusioncache.Signal{}, rejected("prefix was not frozen")
	}
	example := p.job.expected.Examples[reference.example]
	target := example.PromptTokens + reference.row
	if request.TargetIndex != target || !slices.Equal(request.Tokens, example.TeacherTokens[:target]) {
		return fusioncache.Signal{}, rejected("request tokens differ from frozen gold")
	}
	rows, exists := p.cache[reference.example]
	if !exists {
		var err error
		rows, err = p.captureExample(ctx, example)
		if err != nil {
			return fusioncache.Signal{}, err
		}
		p.cache[reference.example] = rows
		p.factory.mu.Lock()
		p.factory.stats.CapturedExamples++
		p.factory.mu.Unlock()
	}
	return copySignal(rows[reference.row]), nil
}

func (p *nativeProducer) captureExample(ctx context.Context, example fusioncache.Example) ([]fusioncache.Signal, error) {
	tokens := make([]int32, len(example.TeacherTokens))
	for i, token := range example.TeacherTokens {
		tokens[i] = int32(token)
	}
	positions := make([]int32, len(tokens)-example.PromptTokens)
	rows := make([]fusioncache.Signal, len(positions))
	for i := range positions {
		target := example.PromptTokens + i
		positions[i] = int32(target - 1)
		rows[i] = fusioncache.Signal{PrefixSHA256: fusioncache.PrefixDigest(example.TeacherTokens[:target]), TargetIndex: target,
			Features: make([]fusioncache.Feature, len(p.job.expected.Features))}
	}
	tensors := []string{}
	for _, spec := range p.job.expected.Features {
		if !slices.Contains(tensors, spec.Tensor) {
			tensors = append(tensors, spec.Tensor)
		}
	}
	if len(tensors) == 0 {
		tensors = append(tensors, p.job.config.Capture.Tensor)
	}
	for tensorIndex, tensor := range tensors {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		options := p.job.config.Capture
		options.Tensor = tensor
		p.factory.mu.Lock()
		p.factory.stats.NativeCaptures++
		p.factory.mu.Unlock()
		captured, err := p.model.CaptureTeacherForced(tokens, positions, options)
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if captured.Version != llama.TeacherCaptureVersion || captured.ModelSHA256 != p.job.expected.Teacher.WeightsSHA256 ||
			captured.Vocabulary != p.job.expected.Teacher.Vocabulary || captured.Tensor != tensor || captured.DType != "f32" ||
			captured.TokenDigest != llama.TeacherTokenDigest(tokens, positions) || len(captured.Rows) != len(rows) {
			return nil, rejected("native capture identity differs")
		}
		for i, native := range captured.Rows {
			if native.Position != positions[i] || native.GoldNextToken != tokens[positions[i]+1] ||
				len(native.TopK) != options.TopK || len(native.Features) != captured.Width {
				return nil, rejected("native capture row differs")
			}
			row := &rows[i]
			if tensorIndex == 0 {
				row.RetainedMass = native.RetainedMass
				row.Probabilities = make([]fusioncache.TeacherProbability, len(native.TopK))
				for k, probability := range native.TopK {
					row.Probabilities[k] = fusioncache.TeacherProbability{TokenID: int64(probability.TokenID), Probability: probability.Probability}
				}
			} else {
				if row.RetainedMass != native.RetainedMass {
					return nil, rejected("feature selection changed teacher distribution")
				}
				for k, probability := range native.TopK {
					if row.Probabilities[k].TokenID != int64(probability.TokenID) || row.Probabilities[k].Probability != probability.Probability {
						return nil, rejected("feature selection changed teacher distribution")
					}
				}
			}
			for featureIndex, spec := range p.job.expected.Features {
				if spec.Tensor != tensor {
					continue
				}
				if spec.Dimension != captured.Width {
					return nil, rejected("native feature dimension differs")
				}
				feature := fusioncache.Feature{Spec: spec, Values: make([]float64, captured.Width)}
				for column, value := range native.Features {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						return nil, rejected("nonfinite native feature")
					}
					feature.Values[column] = float64(value)
				}
				row.Features[featureIndex] = feature
			}
		}
	}
	return rows, nil
}

func copySignal(source fusioncache.Signal) fusioncache.Signal {
	result := source
	result.Probabilities = slices.Clone(source.Probabilities)
	result.Features = make([]fusioncache.Feature, len(source.Features))
	for i, feature := range source.Features {
		result.Features[i] = fusioncache.Feature{Spec: feature.Spec, Values: slices.Clone(feature.Values)}
	}
	return result
}

func (p *nativeProducer) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return p.closeErr
	}
	p.closed = true
	p.closeErr = p.model.Close()
	p.model, p.cache = nil, nil
	p.factory.mu.Lock()
	defer p.factory.mu.Unlock()
	if p.closeErr != nil {
		p.factory.closed = true
		p.factory.closeErr = errors.Join(p.factory.closeErr, p.closeErr)
	} else {
		p.factory.stats.ClosedTeachers++
		p.factory.stats.ResidentTeachers = 0
		p.factory.active = nil
	}
	return p.closeErr
}
