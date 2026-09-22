package support

// Htmlable is a value that knows how to present itself as HTML which must not
// be escaped again.
type Htmlable interface {
	// ToHtml returns the value as markup, ready to write into a template.
	ToHtml() string
}

// Jsonable is a value that knows how to present itself as JSON.
type Jsonable interface {
	// ToJson returns the value as JSON, or the error encoding it raised.
	ToJson() (string, error)
}

// HtmlString is a string already safe to write into a template, which the view
// layer must not escape.
type HtmlString struct {
	html string
}

// NewHtmlString marks the given markup as already escaped. The empty string is
// allowed.
func NewHtmlString(html string) *HtmlString { return &HtmlString{html: html} }

// ToHtml returns the markup. A nil receiver returns the empty string.
func (h *HtmlString) ToHtml() string {
	if h == nil {
		return ""
	}
	return h.html
}

// IsEmpty reports whether the markup is the empty string.
func (h *HtmlString) IsEmpty() bool { return h.ToHtml() == "" }

// IsNotEmpty reports whether the markup holds anything.
func (h *HtmlString) IsNotEmpty() bool { return !h.IsEmpty() }

// String returns the markup, so HtmlString satisfies fmt.Stringer.
func (h *HtmlString) String() string { return h.ToHtml() }
