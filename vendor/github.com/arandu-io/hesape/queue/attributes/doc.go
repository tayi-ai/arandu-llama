// Package attributes is the per-job settings a job carries: how many tries,
// how long to wait between them, how long the handler may run.
//
// They are fields of one [Attributes] struct rather than something read off a
// type by reflection. [Attributes] documents that choice and the reason on the
// type itself, where somebody reading the field will find it.
//
// The package depends on nothing but time, so the job record, the drivers and
// the worker can all read it without importing each other.
package attributes
