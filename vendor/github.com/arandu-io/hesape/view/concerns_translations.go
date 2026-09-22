package view

import "strings"

// The @lang directive captures a translation block; the compiler calls
// StartTranslation to open the block and RenderTranslation to evaluate it.

// translationReplacements holds the replacement pairs active during a
// translation block capture.
type translationReplacements map[string]string

// StartTranslation begins capturing a translation block.
func (f *Factory) StartTranslation(replacements map[string]string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.transReplacements = replacements
}

// RenderTranslation evaluates the captured translation: it reads the key
// from the captured buffer and looks it up.
func (f *Factory) RenderTranslation(key string) string {
	f.mu.RLock()
	replacements := f.transReplacements
	f.mu.RUnlock()

	result := key
	for placeholder, value := range replacements {
		result = strings.ReplaceAll(result, ":"+placeholder, value)
	}
	return result
}

// SetTranslationBuffer stores the captured key for a translation block.
func (f *Factory) SetTranslationBuffer(key string) {
	f.mu.Lock()
	f.transKey = key
	f.mu.Unlock()
}

// GetTranslationBuffer returns the captured translation key.
func (f *Factory) GetTranslationBuffer() string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.transKey
}
