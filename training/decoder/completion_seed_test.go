package decoder

import (
	"math"
	"testing"
)

func completionReferenceLoss(logits []float32, vocabulary int, targets []int64) float64 {
	total := 0.0
	for row, target := range targets {
		offset := row * vocabulary
		maximum := float64(logits[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(logits[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(logits[offset+column]) - maximum)
		}
		total += maximum + math.Log(sum) - float64(logits[offset+int(target)])
	}
	return total / float64(len(targets))
}

func TestCompletionCotangentMatchesReferenceAndFiniteDifference(t *testing.T) {
	original := []float32{
		0.4, -0.3, 1.1, 0.2,
		-0.7, 0.9, 0.1, 0.5,
		9, 8, 7, 6,
	}
	targets := []int64{2, 1}
	got := append([]float32(nil), original...)
	loss, err := completionCotangent(got, 4, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss := completionReferenceLoss(original, 4, targets)
	if math.Abs(loss-wantLoss) > 1e-12 {
		t.Fatalf("loss %.12g != reference %.12g", loss, wantLoss)
	}
	for row := range targets {
		sum := 0.0
		for column := 0; column < 4; column++ {
			sum += float64(got[row*4+column])
		}
		if math.Abs(sum) > 1e-7 {
			t.Fatalf("row %d cotangent sum %.9g", row, sum)
		}
	}
	for _, value := range got[8:] {
		if value != 0 {
			t.Fatal("unscored final row has a cotangent")
		}
	}

	epsilon := float32(1e-3)
	for coordinate := 0; coordinate < 8; coordinate++ {
		plus := append([]float32(nil), original...)
		minus := append([]float32(nil), original...)
		plus[coordinate] += epsilon
		minus[coordinate] -= epsilon
		numerical := (completionReferenceLoss(plus, 4, targets) - completionReferenceLoss(minus, 4, targets)) /
			float64(2*epsilon)
		analytic := float64(got[coordinate])
		if math.Abs(analytic-numerical) > 2e-5 {
			t.Fatalf("coordinate %d analytic %.9g numerical %.9g", coordinate, analytic, numerical)
		}
	}
}

func TestCompletionCotangentScalesOnlyDerivative(t *testing.T) {
	original := []float32{
		0.4, -0.3, 1.1, 0.2,
		-0.7, 0.9, 0.1, 0.5,
		9, 8, 7, 6,
	}
	targets := []int64{2, 1}
	one := append([]float32(nil), original...)
	two := append([]float32(nil), original...)
	lossOne, err := completionCotangent(one, 4, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	lossTwo, err := completionCotangent(two, 4, targets, 2)
	if err != nil {
		t.Fatal(err)
	}
	if lossOne != lossTwo {
		t.Fatalf("loss scale changed loss: %.12g != %.12g", lossOne, lossTwo)
	}
	for index := range one {
		want := 2 * float64(one[index])
		if math.Abs(float64(two[index])-want) > 1e-7 {
			t.Fatalf("scaled cotangent %d %.9g != %.9g", index, two[index], want)
		}
	}
}

func TestCompletionCotangentRejectsInvalidInputs(t *testing.T) {
	valid := []float32{0, 1, 2, 0, 0, 0}
	for name, run := range map[string]func() error{
		"empty targets": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, nil, 1)
			return err
		},
		"wrong geometry": func() error {
			_, err := completionCotangent([]float32{0, 1}, 2, []int64{1}, 1)
			return err
		},
		"target outside vocabulary": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, []int64{2, 0}, 1)
			return err
		},
		"zero scale": func() error {
			_, err := completionCotangent(append([]float32(nil), valid...), 2, []int64{1, 0}, 0)
			return err
		},
		"nonfinite logit": func() error {
			bad := append([]float32(nil), valid...)
			bad[0] = float32(math.NaN())
			_, err := completionCotangent(bad, 2, []int64{1, 0}, 1)
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := run(); err == nil {
				t.Fatal("invalid completion cotangent accepted")
			}
		})
	}
}
