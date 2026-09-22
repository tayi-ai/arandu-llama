package toon

import "fmt"

const (
	defaultDelimiter  = ','
	defaultIndentSize = 2
)

type encodeOptions struct {
	delimiter  byte
	indentSize int
}

// EncodeOption configures Encode.
type EncodeOption func(*encodeOptions) error

// WithDelimiter sets the delimiter used by array headers, table schemas,
// inline arrays, table rows, and delimiter-aware string quoting. The supported
// delimiters are comma, tab, and pipe.
func WithDelimiter(delimiter string) EncodeOption {
	return func(options *encodeOptions) error {
		switch delimiter {
		case ",", "\t", "|":
			options.delimiter = delimiter[0]
			return nil
		default:
			return fmt.Errorf("toon: delimiter %q is not supported; use comma, tab, or pipe", delimiter)
		}
	}
}

// WithIndent sets the positive number of spaces used for each structural
// indentation level.
func WithIndent(size int) EncodeOption {
	return func(options *encodeOptions) error {
		if size <= 0 {
			return fmt.Errorf("toon: indent must be greater than zero")
		}
		options.indentSize = size
		return nil
	}
}

func defaultEncodeOptions() encodeOptions {
	return encodeOptions{
		delimiter:  defaultDelimiter,
		indentSize: defaultIndentSize,
	}
}

func resolveEncodeOptions(options []EncodeOption) (encodeOptions, error) {
	resolved := defaultEncodeOptions()
	for i, option := range options {
		if option == nil {
			return encodeOptions{}, fmt.Errorf("toon: encode option %d is nil", i+1)
		}
		if err := option(&resolved); err != nil {
			return encodeOptions{}, err
		}
	}
	return resolved, nil
}
