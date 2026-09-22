package image

import "github.com/arandu-io/hesape/image/transformations"

// ImagePipeline is the transformations to run, in the order they were asked
// for, and what to encode the result as.
//
// It is the whole of what [Image] accumulates. Nothing here touches a pixel --
// the pipeline is a plan, and the [Driver] is what executes it, once, when
// somebody finally asks for bytes.
type ImagePipeline struct {
	// Transformations are the transformations to run, in order.
	Transformations []transformations.Transformation
	// Output is a value and not a pointer: a pipeline that shared its options
	// with the pipeline it was cloned from would let a quality set on one
	// image reach another, which is exactly what cloning the pipeline is
	// written to prevent.
	Output ImageOutputOptions
}

// NewImagePipeline returns an empty pipeline.
func NewImagePipeline() *ImagePipeline { return &ImagePipeline{} }

// Add appends a transformation to the pipeline.
func (p *ImagePipeline) Add(t transformations.Transformation) {
	p.Transformations = append(p.Transformations, t)
}

// HasChanges reports whether anything at all was asked of this image.
func (p *ImagePipeline) HasChanges() bool {
	return len(p.Transformations) > 0 || p.Output.HasChanges()
}

// clone returns a copy of the pipeline. The slice is copied rather than
// shared: two images that grew from one call to Blur must not append into the
// same backing array.
func (p *ImagePipeline) clone() *ImagePipeline {
	out := &ImagePipeline{Output: p.Output}
	if len(p.Transformations) > 0 {
		out.Transformations = make([]transformations.Transformation, len(p.Transformations))
		copy(out.Transformations, p.Transformations)
	}
	return out
}
