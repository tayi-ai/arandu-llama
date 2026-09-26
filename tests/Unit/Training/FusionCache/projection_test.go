package fusioncache_test

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
)

func projectionLimits() fusioncache.ProjectionLimits {
	return fusioncache.ProjectionLimits{MaxSamples: 16, MaxDimension: 8192, MaxElements: 25000}
}

func sample(id string, source, target []float64) fusioncache.FeatureSample {
	return fusioncache.FeatureSample{Identity: fusioncache.SampleIdentity{DatasetSHA256: strings.Repeat("1", 64), ExampleID: id, Role: "calibration", TargetIndex: 2, TeacherPrefixSHA256: fusioncache.PrefixDigest([]int64{1, 2}), StudentPrefixSHA256: fusioncache.PrefixDigest([]int64{1, 2})}, Source: source, Target: target}
}

func matchingPlan(fit, heldout []fusioncache.FeatureSample) fusioncache.MatchingPlan {
	p := fusioncache.MatchingPlan{ID: "frozen-layer-match", SourceModel: model("teacher"), TargetModel: model("student"), TokenMappingSHA256: strings.Repeat("9", 64), Source: fusioncache.FeatureSpec{Layer: 3, Tensor: "post_norm", DType: "float32", Dimension: len(fit[0].Source)}, Target: fusioncache.FeatureSpec{Layer: 1, Tensor: "post_norm", DType: "float32", Dimension: len(fit[0].Target)}, Ridge: 0.5}
	for _, s := range fit {
		p.Fit = append(p.Fit, s.Identity)
	}
	for _, s := range heldout {
		p.Heldout = append(p.Heldout, s.Identity)
	}
	return p
}

func TestRidgeMatchesIndependentKnownTransform(t *testing.T) {
	fit := []fusioncache.FeatureSample{sample("fit-one", []float64{1, 0}, []float64{2, 3}), sample("fit-two", []float64{0, 1}, []float64{-1, 4})}
	heldout := []fusioncache.FeatureSample{sample("heldout", []float64{2, 1}, []float64{3, 10})}
	plan := matchingPlan(fit, heldout)
	sha := digest(t, plan)
	projection, report, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	// X is identity, so ridge gives P = W/(1+lambda), independently of the solver.
	want := [][]float64{{4.0 / 3, 2}, {-2.0 / 3, 8.0 / 3}}
	for i, row := range projection.Coefficients {
		for j, value := range row {
			if math.Abs(value-want[i][j]) > 1e-12 {
				t.Fatalf("coefficient %d/%d got %.16g want %.16g", i, j, value, want[i][j])
			}
		}
	}
	got, err := projection.Apply([]float64{2, 1}, sha, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got[0]-2) > 1e-12 || math.Abs(got[1]-20.0/3) > 1e-12 {
		t.Fatalf("unexpected projection %v", got)
	}
	if report.Dual || report.FitValues != 4 || report.HeldoutValues != 2 || math.Abs(report.HeldoutSquaredError-(1+100.0/9)) > 1e-12 {
		t.Fatalf("incorrect report %+v", report)
	}
	again, againReport, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil || !reflect.DeepEqual(projection, again) || report != againReport {
		t.Fatalf("fit is not deterministic: %v", err)
	}
}

func TestDualRidgeFitsWideFeaturesWithinMemoryBudget(t *testing.T) {
	first, second, validation := make([]float64, 4096), make([]float64, 4096), make([]float64, 4096)
	first[0], second[1], validation[0], validation[1] = 1, 1, 2, 1
	fit := []fusioncache.FeatureSample{sample("fit-one", first, []float64{2, 3}), sample("fit-two", second, []float64{-1, 4})}
	heldout := []fusioncache.FeatureSample{sample("heldout", validation, []float64{3, 10})}
	plan := matchingPlan(fit, heldout)
	sha := digest(t, plan)
	p, report, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if !p.Dual || !report.Dual || len(p.Basis) != 2 || len(p.Coefficients) != 2 {
		t.Fatalf("wide projection materialized instead of dual: %+v", report)
	}
	got, err := p.Apply(validation, sha, projectionLimits())
	if err != nil || math.Abs(got[0]-2) > 1e-12 || math.Abs(got[1]-20.0/3) > 1e-12 {
		t.Fatalf("dual differs from independent ridge: %v %v", got, err)
	}
	bound := projectionLimits()
	bound.MaxElements = 10
	if _, _, err := fusioncache.FitProjection(plan, sha, fit, heldout, bound); err == nil {
		t.Fatal("allocation budget ignored")
	}
}

func TestRidgeRegularizesRankDeficientFeatures(t *testing.T) {
	fit := []fusioncache.FeatureSample{sample("fit-one", []float64{1, 2}, []float64{3}), sample("fit-two", []float64{2, 4}, []float64{6})}
	heldout := []fusioncache.FeatureSample{sample("heldout", []float64{3, 6}, []float64{9})}
	plan := matchingPlan(fit, heldout)
	plan.Ridge = 1
	p, _, err := fusioncache.FitProjection(plan, digest(t, plan), fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p.Coefficients[0][0]-15.0/26) > 1e-12 || math.Abs(p.Coefficients[1][0]-30.0/26) > 1e-12 {
		t.Fatalf("wrong rank deficient solution: %v", p.Coefficients)
	}
	plan.Ridge = 0
	if _, _, err := fusioncache.FitProjection(plan, digest(t, plan), fit, heldout, projectionLimits()); err == nil {
		t.Fatal("unregularized fit admitted")
	}
}

func TestHeldoutValuesNeverAffectFittedProjection(t *testing.T) {
	fit := []fusioncache.FeatureSample{sample("fit", []float64{1, 0}, []float64{3})}
	heldout := []fusioncache.FeatureSample{sample("heldout", []float64{2, 0}, []float64{6})}
	plan := matchingPlan(fit, heldout)
	sha := digest(t, plan)
	first, r1, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	heldout[0].Target[0] = 999
	second, r2, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) || r1.HeldoutSquaredError == r2.HeldoutSquaredError {
		t.Fatal("heldout labels affected fitting or were not evaluated")
	}
}

func TestProjectionRejectsChangedPlanReservedDataAndNonfinite(t *testing.T) {
	fit := []fusioncache.FeatureSample{sample("fit", []float64{1, 0}, []float64{3})}
	heldout := []fusioncache.FeatureSample{sample("heldout", []float64{2, 0}, []float64{6})}
	plan := matchingPlan(fit, heldout)
	sha := digest(t, plan)
	changed := plan
	changed.Source.Layer++
	if _, _, err := fusioncache.FitProjection(changed, sha, fit, heldout, projectionLimits()); err == nil {
		t.Fatal("post hoc layer matching accepted")
	}
	for _, role := range []string{"protection", "sealed-final", "selection"} {
		badFit := clone(t, fit)
		badFit[0].Identity.Role = role
		badPlan := matchingPlan(badFit, heldout)
		if _, _, err := fusioncache.FitProjection(badPlan, digest(t, badPlan), badFit, heldout, projectionLimits()); err == nil {
			t.Fatalf("fit consumed %s", role)
		}
	}
	leaked := clone(t, heldout)
	leaked[0].Identity.ExampleID = fit[0].Identity.ExampleID
	leaked[0].Identity.TargetIndex++
	badPlan := matchingPlan(fit, leaked)
	if _, _, err := fusioncache.FitProjection(badPlan, digest(t, badPlan), fit, leaked, projectionLimits()); err == nil {
		t.Fatal("same example leaked between fit and heldout at different positions")
	}
	for _, value := range []float64{math.NaN(), math.Inf(1), math.MaxFloat64} {
		badFit := clone(t, fit)
		badFit[0].Source[0] = value
		if _, _, err := fusioncache.FitProjection(plan, sha, badFit, heldout, projectionLimits()); err == nil {
			t.Fatalf("nonfinite input or overflowing solve admitted: %g", value)
		}
	}
	p, _, err := fusioncache.FitProjection(plan, sha, fit, heldout, projectionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Apply([]float64{math.NaN(), 0}, sha, projectionLimits()); err == nil {
		t.Fatal("nonfinite application input admitted")
	}
	p.Coefficients[0][0]++
	if _, err := p.Apply([]float64{1, 0}, sha, projectionLimits()); err == nil {
		t.Fatal("mutated projection admitted")
	}
}
