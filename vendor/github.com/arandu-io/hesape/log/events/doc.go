// Package events is the event a written log line fires.
//
// There is one, MessageLogged, and it carries the level, the formatted message
// and the merged context of the line.
//
// The package is separate from log because a listener imports the event, and
// importing the event should not drag in the logger.
package events
