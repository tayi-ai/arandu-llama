package jobs

import "strings"

// JobName reads the several names a job has.
//
// It is a struct with no fields: the three methods belong together and none of
// them needs a receiver, so the type is a name for the reading and
// `jobs.JobName{}.Resolve(...)` is all there is to it.
//
// A job has three names because a wrapper has two:
//
//	Name          what routes it to a handler
//	DisplayName   what a person reads
//	class name    what is really behind a wrapper
//
// For an ordinary job all three are the same string, and these three methods
// are what say so in one place instead of at every call site.
type JobName struct{}

// Parse splits a job name into the name and the method behind it.
//
// A name here is "invoice.send" and there is no method to call, so a name with
// no "@" comes back with the default method "fire".
//
// It is kept because a record written by an older release, or by another system
// pushing onto the same store, can still carry the "name@method" form, and
// routing "SendInvoice@handle" to a handler registered as "SendInvoice" is
// better than failing to route it at all.
func (JobName) Parse(name string) (string, string) {
	if class, method, found := strings.Cut(name, "@"); found {
		return class, method
	}
	return name, "fire"
}

// Resolve is the name to show for a job.
//
// It prefers the job's DisplayName. A nil job, or one with no DisplayName,
// answers with name.
func (JobName) Resolve(name string, j *Job) string {
	if j != nil && j.DisplayName != "" {
		return j.DisplayName
	}
	return name
}

// ResolveClassName is the name of the work behind a wrapper.
//
// The name that routes is the name that was registered, so for a job that wraps
// nothing this is that name.
func (JobName) ResolveClassName(name string, j *Job) string {
	if j != nil && j.Name != "" {
		return j.Name
	}
	class, _ := JobName{}.Parse(name)
	return class
}
