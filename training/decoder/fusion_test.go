package decoder

import (
	"math"
	"testing"
)

func fusionFixture() ([]float32, []int64, []FusionTeacher) {
	logits := []float32{
		0.4, -0.3, 1.1, 0.2,
		-0.7, 0.9, 0.1, 0.5,
		9, 8, 7, 6,
	}
	targets := []int64{2, 1}
	teachers := []FusionTeacher{
		{
			Name: "teacher-a", Weight: 0.30,
			Positions: []FusionTeacherPosition{
				{RetainedMass: 0.90, TopK: []FusionTokenProbability{{2, 0.60}, {1, 0.30}}},
				{RetainedMass: 1.00, TopK: []FusionTokenProbability{{1, 0.80}, {0, 0.20}}},
			},
		},
		{
			Name: "teacher-b", Weight: 0.20,
			Positions: []FusionTeacherPosition{
				{RetainedMass: 0.80, TopK: []FusionTokenProbability{{2, 0.50}, {3, 0.30}}},
				{RetainedMass: 0.70, TopK: []FusionTokenProbability{{1, 0.40}, {2, 0.30}}},
			},
		},
	}
	return logits, targets, teachers
}
func softmaxRow(values []float32) []float64 {
	maximum := float64(values[0])
	for _, value := range values[1:] {
		maximum = math.Max(maximum, float64(value))
	}
	out := make([]float64, len(values))
	sum := 0.0
	for i, value := range values {
		out[i] = math.Exp(float64(value) - maximum)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func manualFusionLoss(logits []float32) float64 {
	targets := [][]float64{
		{0, 0.09, 0.85, 0.06},
		{0.06, 0.88, 0.06, 0},
	}
	total := 0.0
	for row := range targets {
		p := softmaxRow(logits[row*4 : row*4+4])
		for token, mass := range targets[row] {
			if mass != 0 {
				total -= mass * math.Log(p[token])
			}
		}
	}
	return total / 2
}
func TestFusionCotangentMatchesBlendedTargetAndFiniteDifference(t *testing.T) {
	original, targets, teachers := fusionFixture()
	got := append([]float32(nil), original...)
	stats, err := fusionCotangent(got, 4, targets, 1, teachers)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss := manualFusionLoss(original)
	if math.Abs(stats.loss-wantLoss) > 1e-10 {
		t.Fatalf("loss %.12g != %.12g", stats.loss, wantLoss)
	}
	if math.Abs(stats.effectiveMass["teacher-a"]-0.285) > 1e-12 ||
		math.Abs(stats.effectiveMass["teacher-b"]-0.15) > 1e-12 {
		t.Fatalf("effective mass: %#v", stats.effectiveMass)
	}
	for row := 0; row < 2; row++ {
		sum := 0.0
		for column := 0; column < 4; column++ {
			sum += float64(got[row*4+column])
		}
		if math.Abs(sum) > 1e-7 {
			t.Fatalf("row %d gradient sum %.9g", row, sum)
		}
	}
	for _, value := range got[8:] {
		if value != 0 {
			t.Fatal("unscored final row is not zero")
		}
	}
	epsilon := float32(1e-3)
	for coordinate := 0; coordinate < 8; coordinate++ {
		plus := append([]float32(nil), original...)
		minus := append([]float32(nil), original...)
		plus[coordinate] += epsilon
		minus[coordinate] -= epsilon
		numerical := (manualFusionLoss(plus) - manualFusionLoss(minus)) / float64(2*epsilon)
		analytic := float64(got[coordinate])
		if math.Abs(analytic-numerical) > 2e-5 {
			t.Fatalf("coordinate %d analytic %.9g numerical %.9g", coordinate, analytic, numerical)
		}
	}
}

func TestFusionCotangentLossScaleOnlyScalesGradient(t *testing.T) {
	original, targets, teachers := fusionFixture()
	one := append([]float32(nil), original...)
	two := append([]float32(nil), original...)
	a, err := fusionCotangent(one, 4, targets, 1, teachers)
	if err != nil {
		t.Fatal(err)
	}
	b, err := fusionCotangent(two, 4, targets, 2, teachers)
	if err != nil {
		t.Fatal(err)
	}
	if a.loss != b.loss || a.hardLoss != b.hardLoss {
		t.Fatal("loss scale changed reported losses")
	}
	for i := range one {
		if math.Abs(float64(two[i])-2*float64(one[i])) > 1e-7 {
			t.Fatalf("scaled gradient %d differs", i)
		}
	}
}

func TestFusionCotangentRejectsInvalidTeacherCaches(t *testing.T) {
	original, targets, _ := fusionFixture()
	cases := map[string]func([]FusionTeacher) []FusionTeacher{
		"weights exceed one": func(in []FusionTeacher) []FusionTeacher {
			in[0].Weight, in[1].Weight = 0.8, 0.3
			return in
		},
		"duplicate teacher name": func(in []FusionTeacher) []FusionTeacher {
			in[1].Name = in[0].Name
			return in
		},
		"wrong positions": func(in []FusionTeacher) []FusionTeacher {
			in[0].Positions = in[0].Positions[:1]
			return in
		},
		"retained mass mismatch": func(in []FusionTeacher) []FusionTeacher {
			in[0].Positions[0].RetainedMass = 0.91
			return in
		},
		"token outside vocabulary": func(in []FusionTeacher) []FusionTeacher {
			in[0].Positions[0].TopK[0].TokenID = 4
			return in
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			_, _, base := fusionFixture()
			bad := mutate(base)
			values := append([]float32(nil), original...)
			if _, err := fusionCotangent(values, 4, targets, 1, bad); err == nil {
				t.Fatal("invalid teacher cache accepted")
			}
		})
	}
}

func TestFusionCotangentWithoutTeachersEqualsHardCompletion(t *testing.T) {
	original, targets, _ := fusionFixture()
	fused := append([]float32(nil), original...)
	hard := append([]float32(nil), original...)
	stats, err := fusionCotangent(fused, 4, targets, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := completionCotangent(hard, 4, targets, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(stats.loss-loss) > 1e-12 {
		t.Fatalf("fusion hard loss %.12g != completion %.12g", stats.loss, loss)
	}
	for i := range fused {
		if math.Abs(float64(fused[i]-hard[i])) > 1e-7 {
			t.Fatalf("hard gradient differs at %d", i)
		}
	}
}
