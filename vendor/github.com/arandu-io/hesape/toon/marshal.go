package toon

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

// SpecVersion is the TOON language version implemented by Marshal. The normative
// corpus is pinned separately to the specification repository's v4.1.1 tag.
const SpecVersion = "4.1"

// MediaType is the provisional media type assigned by the TOON specification.
const MediaType = "text/toon"

type valueKind uint8

const (
	kindNull valueKind = iota
	kindBool
	kindNumber
	kindString
	kindObject
	kindArray
)

type member struct {
	name  string
	value node
}

type node struct {
	kind   valueKind
	text   string
	object []member
	array  []node
}

type fieldSchema struct {
	name   string
	fields []fieldSchema
}

// Encode encodes value as a TOON 4.1 string using the supplied options.
// Without options, Encode uses the same canonical comma and two-space profile
// as Marshal.
func Encode(value any, options ...EncodeOption) (string, error) {
	resolved, err := resolveEncodeOptions(options)
	if err != nil {
		return "", err
	}
	return encode(value, resolved)
}

// Marshal encodes value as TOON 4.1 from the pinned v4.1.1 specification
// release using the fixed comma and two-space canonical profile.
//
// Marshal first applies encoding/json's field, tag, omission, and Marshaler
// rules. Values encoding/json refuses are refused here too. Object keys from Go
// maps therefore have encoding/json's deterministic lexical order; struct and
// custom-marshaled objects keep the order of their JSON representation.
// Malformed UTF-8, unpaired surrogates, duplicate object keys, and values nested
// more than 1,000 levels are refused rather than silently changed. Non-JSON Go
// values, including cycles, channels, functions, NaN, and infinities, are
// outside Marshal's input domain and return an error.
func Marshal(value any) ([]byte, error) {
	encoded, err := encode(value, defaultEncodeOptions())
	if err != nil {
		return nil, err
	}
	return []byte(encoded), nil
}

func encode(value any, options encodeOptions) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("toon: normalize JSON value: %w", err)
	}
	replacedInvalidUTF8, err := validateJSONStrings(encoded)
	if err != nil {
		return "", err
	}

	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	root, err := decodeNode(decoder, 0)
	if err != nil {
		return "", fmt.Errorf("toon: normalize JSON value: %w", err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		if err == nil {
			return "", fmt.Errorf("toon: normalize JSON value: trailing JSON value")
		}
		return "", fmt.Errorf("toon: normalize JSON value: %w", err)
	}
	if replacedInvalidUTF8 {
		if err := validateInputStrings(reflect.ValueOf(value), make(map[visit]struct{})); err != nil {
			return "", err
		}
	}

	var lines []string
	if err := appendNode(&lines, root, 0, options); err != nil {
		return "", err
	}
	return strings.Join(lines, "\n"), nil
}

const maxNestingDepth = 1000

func decodeNode(decoder *json.Decoder, depth int) (node, error) {
	token, err := decoder.Token()
	if err != nil {
		return node{}, err
	}

	switch token := token.(type) {
	case nil:
		return node{kind: kindNull}, nil
	case bool:
		return node{kind: kindBool, text: strconv.FormatBool(token)}, nil
	case json.Number:
		return node{kind: kindNumber, text: canonicalNumber(token.String())}, nil
	case string:
		return node{kind: kindString, text: token}, nil
	case json.Delim:
		if depth >= maxNestingDepth {
			return node{}, fmt.Errorf("nesting exceeds %d levels", maxNestingDepth)
		}
		switch token {
		case '{':
			return decodeObject(decoder, depth+1)
		case '[':
			return decodeArray(decoder, depth+1)
		default:
			return node{}, fmt.Errorf("unexpected delimiter %q", token)
		}
	default:
		return node{}, fmt.Errorf("unsupported JSON token %T", token)
	}
}

func decodeObject(decoder *json.Decoder, depth int) (node, error) {
	object := make([]member, 0)
	names := make(map[string]struct{})
	for decoder.More() {
		nameToken, err := decoder.Token()
		if err != nil {
			return node{}, err
		}
		name, ok := nameToken.(string)
		if !ok {
			return node{}, fmt.Errorf("object key is %T, want string", nameToken)
		}
		if _, duplicate := names[name]; duplicate {
			return node{}, fmt.Errorf("duplicate object key %q", name)
		}
		names[name] = struct{}{}
		value, err := decodeNode(decoder, depth)
		if err != nil {
			return node{}, err
		}
		object = append(object, member{name: name, value: value})
	}
	if _, err := decoder.Token(); err != nil {
		return node{}, err
	}
	return node{kind: kindObject, object: object}, nil
}

func decodeArray(decoder *json.Decoder, depth int) (node, error) {
	array := make([]node, 0)
	for decoder.More() {
		value, err := decodeNode(decoder, depth)
		if err != nil {
			return node{}, err
		}
		array = append(array, value)
	}
	if _, err := decoder.Token(); err != nil {
		return node{}, err
	}
	return node{kind: kindArray, array: array}, nil
}

func appendNode(lines *[]string, value node, depth int, options encodeOptions) error {
	switch value.kind {
	case kindObject:
		if schema, ok := keyedSchema(value.object); ok {
			appendKeyedTable(lines, "", value.object, schema, depth+1, options)
			return nil
		}
		for _, field := range value.object {
			if err := appendField(lines, field, indent(depth, options.indentSize), depth+1, options); err != nil {
				return err
			}
		}
		return nil
	case kindArray:
		if len(value.array) == 0 {
			*lines = append(*lines, "[]")
			return nil
		}
		if allPrimitive(value.array) {
			*lines = append(*lines, arrayHeader(len(value.array), false, options.delimiter)+": "+joinPrimitives(value.array, options.delimiter))
			return nil
		}
		if schema, ok := tabularSchema(value.array); ok {
			appendTable(lines, "", value.array, schema, 1, options)
			return nil
		}
		return appendListArray(lines, "", value.array, 1, options)
	case kindNull, kindBool, kindNumber, kindString:
		*lines = append(*lines, indent(depth, options.indentSize)+encodePrimitive(value, options.delimiter))
		return nil
	default:
		return fmt.Errorf("toon: value kind %d is not implemented", value.kind)
	}
}

func appendField(lines *[]string, field member, linePrefix string, childDepth int, options encodeOptions) error {
	key := encodeKey(field.name)
	if isPrimitive(field.value) {
		*lines = append(*lines, linePrefix+key+": "+encodePrimitive(field.value, options.delimiter))
		return nil
	}
	if field.value.kind == kindArray && allPrimitive(field.value.array) {
		if len(field.value.array) == 0 {
			*lines = append(*lines, linePrefix+key+": []")
			return nil
		}
		*lines = append(*lines, linePrefix+key+arrayHeader(len(field.value.array), false, options.delimiter)+": "+joinPrimitives(field.value.array, options.delimiter))
		return nil
	}
	if field.value.kind == kindArray {
		if schema, ok := tabularSchema(field.value.array); ok {
			appendTable(lines, linePrefix+key, field.value.array, schema, childDepth, options)
			return nil
		}
		return appendListArray(lines, linePrefix+key, field.value.array, childDepth, options)
	}
	if field.value.kind == kindObject {
		if schema, ok := keyedSchema(field.value.object); ok {
			appendKeyedTable(lines, linePrefix+key, field.value.object, schema, childDepth, options)
			return nil
		}
		*lines = append(*lines, linePrefix+key+":")
		for _, nested := range field.value.object {
			if err := appendField(lines, nested, indent(childDepth, options.indentSize), childDepth+1, options); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("toon: unsupported value at %q", field.name)
}

func appendListArray(lines *[]string, headerPrefix string, values []node, itemDepth int, options encodeOptions) error {
	*lines = append(*lines, headerPrefix+arrayHeader(len(values), false, options.delimiter)+":")
	for _, value := range values {
		if err := appendListItem(lines, value, itemDepth, options); err != nil {
			return err
		}
	}
	return nil
}

func appendListItem(lines *[]string, value node, depth int, options encodeOptions) error {
	prefix := indent(depth, options.indentSize) + "- "
	switch value.kind {
	case kindNull, kindBool, kindNumber, kindString:
		*lines = append(*lines, prefix+encodePrimitive(value, options.delimiter))
	case kindArray:
		if len(value.array) == 0 {
			*lines = append(*lines, prefix+arrayHeader(0, false, options.delimiter)+":")
			return nil
		}
		if allPrimitive(value.array) {
			*lines = append(*lines, prefix+arrayHeader(len(value.array), false, options.delimiter)+": "+joinPrimitives(value.array, options.delimiter))
			return nil
		}
		*lines = append(*lines, prefix+arrayHeader(len(value.array), false, options.delimiter)+":")
		for _, nested := range value.array {
			if err := appendListItem(lines, nested, depth+1, options); err != nil {
				return err
			}
		}
	case kindObject:
		if len(value.object) == 0 {
			*lines = append(*lines, strings.TrimSuffix(prefix, " "))
			return nil
		}
		if err := appendField(lines, value.object[0], prefix, depth+2, options); err != nil {
			return err
		}
		for _, field := range value.object[1:] {
			if err := appendField(lines, field, indent(depth+1, options.indentSize), depth+2, options); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("toon: unsupported list value kind %d", value.kind)
	}
	return nil
}

func tabularSchema(values []node) ([]fieldSchema, bool) {
	if len(values) == 0 {
		return nil, false
	}
	for _, value := range values {
		if value.kind != kindObject || len(value.object) == 0 {
			return nil, false
		}
	}
	return schemaForObjects(values)
}

func keyedSchema(object []member) ([]fieldSchema, bool) {
	if len(object) < 2 {
		return nil, false
	}
	values := make([]node, len(object))
	for i, entry := range object {
		values[i] = entry.value
	}
	return tabularSchema(values)
}

func schemaForObjects(objects []node) ([]fieldSchema, bool) {
	first := objects[0].object
	for _, object := range objects[1:] {
		if len(object.object) != len(first) {
			return nil, false
		}
		for _, field := range first {
			if _, ok := findMember(object.object, field.name); !ok {
				return nil, false
			}
		}
	}

	schema := make([]fieldSchema, 0, len(first))
	for _, field := range first {
		column := make([]node, len(objects))
		for i, object := range objects {
			column[i], _ = findMember(object.object, field.name)
		}

		if allPrimitive(column) {
			schema = append(schema, fieldSchema{name: field.name})
			continue
		}
		allObjects := true
		for _, value := range column {
			if value.kind != kindObject || len(value.object) == 0 {
				allObjects = false
				break
			}
		}
		if !allObjects {
			return nil, false
		}
		nested, ok := schemaForObjects(column)
		if !ok {
			return nil, false
		}
		schema = append(schema, fieldSchema{name: field.name, fields: nested})
	}
	return schema, true
}

func findMember(object []member, name string) (node, bool) {
	for _, field := range object {
		if field.name == name {
			return field.value, true
		}
	}
	return node{}, false
}

func appendTable(lines *[]string, headerPrefix string, values []node, schema []fieldSchema, rowDepth int, options encodeOptions) {
	*lines = append(*lines, headerPrefix+arrayHeader(len(values), false, options.delimiter)+"{"+encodeSchema(schema, options.delimiter)+"}:")
	for _, value := range values {
		cells := make([]node, 0, leafCount(schema))
		appendCells(&cells, value, schema)
		*lines = append(*lines, indent(rowDepth, options.indentSize)+joinPrimitives(cells, options.delimiter))
	}
}

func appendKeyedTable(lines *[]string, headerPrefix string, object []member, schema []fieldSchema, rowDepth int, options encodeOptions) {
	*lines = append(*lines, headerPrefix+arrayHeader(len(object), true, options.delimiter)+"{"+encodeSchema(schema, options.delimiter)+"}:")
	for _, entry := range object {
		cells := make([]node, 0, leafCount(schema))
		appendCells(&cells, entry.value, schema)
		*lines = append(*lines, indent(rowDepth, options.indentSize)+encodeKey(entry.name)+": "+joinPrimitives(cells, options.delimiter))
	}
}

func encodeSchema(schema []fieldSchema, delimiter byte) string {
	fields := make([]string, len(schema))
	for i, field := range schema {
		fields[i] = encodeKey(field.name)
		if len(field.fields) > 0 {
			fields[i] += "{" + encodeSchema(field.fields, delimiter) + "}"
		}
	}
	return strings.Join(fields, string(delimiter))
}

func leafCount(schema []fieldSchema) int {
	count := 0
	for _, field := range schema {
		if len(field.fields) == 0 {
			count++
		} else {
			count += leafCount(field.fields)
		}
	}
	return count
}

func appendCells(cells *[]node, object node, schema []fieldSchema) {
	for _, field := range schema {
		value, _ := findMember(object.object, field.name)
		if len(field.fields) == 0 {
			*cells = append(*cells, value)
		} else {
			appendCells(cells, value, field.fields)
		}
	}
}

func allPrimitive(values []node) bool {
	for _, value := range values {
		if !isPrimitive(value) {
			return false
		}
	}
	return true
}

func joinPrimitives(values []node, delimiter byte) string {
	encoded := make([]string, len(values))
	for i, value := range values {
		encoded[i] = encodePrimitive(value, delimiter)
	}
	return strings.Join(encoded, string(delimiter))
}

func isPrimitive(value node) bool {
	return value.kind == kindNull || value.kind == kindBool || value.kind == kindNumber || value.kind == kindString
}

func encodePrimitive(value node, delimiter byte) string {
	switch value.kind {
	case kindNull:
		return "null"
	case kindBool, kindNumber:
		return value.text
	case kindString:
		return encodeString(value.text, delimiter)
	default:
		panic("toon: encodePrimitive called with a structured value")
	}
}

func encodeKey(key string) string {
	if isUnquotedKey(key) {
		return key
	}
	return quote(key)
}

func isUnquotedKey(key string) bool {
	if key == "" || !isASCIIAlpha(key[0]) && key[0] != '_' {
		return false
	}
	for i := 1; i < len(key); i++ {
		if !isASCIIAlpha(key[i]) && !isASCIIDigit(key[i]) && key[i] != '_' && key[i] != '.' {
			return false
		}
	}
	return true
}

func isASCIIAlpha(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z'
}

func isASCIIDigit(c byte) bool { return c >= '0' && c <= '9' }

func encodeString(value string, delimiter byte) string {
	if mustQuote(value, delimiter) {
		return quote(value)
	}
	return value
}

func mustQuote(value string, delimiter byte) bool {
	if value == "" || value == "true" || value == "false" || value == "null" {
		return true
	}
	if value[0] == '-' || value[0] == '#' || value[0] == ' ' || value[0] == '\t' ||
		value[len(value)-1] == ' ' || value[len(value)-1] == '\t' {
		return true
	}
	for i := 0; i < len(value); i++ {
		if value[i] < 0x20 || value[i] == delimiter || strings.ContainsRune(":\"\\[]{}", rune(value[i])) {
			return true
		}
	}
	return isNumericLike(value)
}

func isNumericLike(value string) bool {
	if value == "" {
		return false
	}
	i := 0
	if value[i] == '+' || value[i] == '-' {
		i++
		if i == len(value) {
			return false
		}
	}
	start := i
	for i < len(value) && isASCIIDigit(value[i]) {
		i++
	}
	if i == start {
		return false
	}
	if i < len(value) && value[i] == '.' {
		i++
		fraction := i
		for i < len(value) && isASCIIDigit(value[i]) {
			i++
		}
		if i == fraction {
			return false
		}
	}
	if i < len(value) && (value[i] == 'e' || value[i] == 'E') {
		i++
		if i < len(value) && (value[i] == '+' || value[i] == '-') {
			i++
		}
		exponent := i
		for i < len(value) && isASCIIDigit(value[i]) {
			i++
		}
		if i == exponent {
			return false
		}
	}
	return i == len(value)
}

func quote(value string) string {
	var out strings.Builder
	out.Grow(len(value) + 2)
	out.WriteByte('"')
	for _, r := range value {
		switch r {
		case '\\':
			out.WriteString(`\\`)
		case '"':
			out.WriteString(`\"`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&out, `\u%04x`, r)
			} else {
				out.WriteRune(r)
			}
		}
	}
	out.WriteByte('"')
	return out.String()
}

func arrayHeader(length int, keyed bool, delimiter byte) string {
	marker := ""
	if delimiter != defaultDelimiter {
		marker = string(delimiter)
	}
	if keyed {
		return fmt.Sprintf("[%d:%s]", length, marker)
	}
	return fmt.Sprintf("[%d%s]", length, marker)
}

func indent(depth, size int) string { return strings.Repeat(" ", depth*size) }

var jsonMarshalerType = reflect.TypeFor[json.Marshaler]()
var textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()

type visit struct {
	typeOf  reflect.Type
	pointer uintptr
}

func validateInputStrings(value reflect.Value, visited map[visit]struct{}) error {
	if !value.IsValid() {
		return nil
	}
	if (value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface) && value.IsNil() {
		return nil
	}
	if value.CanInterface() && value.Type().Implements(jsonMarshalerType) {
		return nil
	}
	if value.CanAddr() && value.Addr().CanInterface() && value.Addr().Type().Implements(jsonMarshalerType) {
		return nil
	}
	if implementsTextMarshaler(value) {
		return fmt.Errorf("toon: cannot verify a JSON UTF-8 replacement produced alongside encoding.TextMarshaler")
	}

	switch value.Kind() {
	case reflect.Interface, reflect.Pointer:
		if value.IsNil() {
			return nil
		}
		if value.Kind() == reflect.Pointer {
			seen := visit{typeOf: value.Type(), pointer: value.Pointer()}
			if _, ok := visited[seen]; ok {
				return nil
			}
			visited[seen] = struct{}{}
			defer delete(visited, seen)
		}
		return validateInputStrings(value.Elem(), visited)
	case reflect.String:
		if !utf8.ValidString(value.String()) {
			return fmt.Errorf("toon: value contains invalid UTF-8")
		}
	case reflect.Struct:
		for i := 0; i < value.NumField(); i++ {
			field := value.Type().Field(i)
			fieldType := field.Type
			if fieldType.Kind() == reflect.Pointer {
				fieldType = fieldType.Elem()
			}
			if field.PkgPath != "" && (!field.Anonymous || fieldType.Kind() != reflect.Struct) {
				continue
			}
			if field.Tag.Get("json") == "-" {
				continue
			}
			if err := validateInputStrings(value.Field(i), visited); err != nil {
				return err
			}
		}
	case reflect.Array:
		for i := 0; i < value.Len(); i++ {
			if err := validateInputStrings(value.Index(i), visited); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if value.IsNil() {
			return nil
		}
		seen := visit{typeOf: value.Type(), pointer: value.Pointer()}
		if _, ok := visited[seen]; ok {
			return nil
		}
		visited[seen] = struct{}{}
		defer delete(visited, seen)
		for i := 0; i < value.Len(); i++ {
			if err := validateInputStrings(value.Index(i), visited); err != nil {
				return err
			}
		}
	case reflect.Map:
		if value.IsNil() {
			return nil
		}
		seen := visit{typeOf: value.Type(), pointer: value.Pointer()}
		if _, ok := visited[seen]; ok {
			return nil
		}
		visited[seen] = struct{}{}
		defer delete(visited, seen)
		iterator := value.MapRange()
		for iterator.Next() {
			if err := validateMapKey(iterator.Key()); err != nil {
				return err
			}
			if err := validateInputStrings(iterator.Value(), visited); err != nil {
				return err
			}
		}
	}
	return nil
}

func implementsTextMarshaler(value reflect.Value) bool {
	if value.CanInterface() && value.Type().Implements(textMarshalerType) {
		return true
	}
	if value.CanAddr() && value.Addr().CanInterface() && value.Addr().Type().Implements(textMarshalerType) {
		return true
	}
	return false
}

func validateMapKey(key reflect.Value) error {
	if key.Kind() == reflect.String {
		if !utf8.ValidString(key.String()) {
			return fmt.Errorf("toon: map key contains invalid UTF-8")
		}
		return nil
	}
	if key.Kind() == reflect.Pointer && key.IsNil() {
		return nil
	}
	if implementsTextMarshaler(key) {
		return fmt.Errorf("toon: cannot verify a JSON UTF-8 replacement produced alongside an encoding.TextMarshaler map key")
	}
	return nil
}

func validateJSONStrings(encoded []byte) (bool, error) {
	if !utf8.Valid(encoded) {
		return false, fmt.Errorf("toon: JSON normalization produced invalid UTF-8")
	}
	replacedInvalidUTF8 := false
	for i := 0; i < len(encoded); i++ {
		if encoded[i] != '"' {
			continue
		}
		i++
		for i < len(encoded) && encoded[i] != '"' {
			if encoded[i] != '\\' {
				i++
				continue
			}
			if i+1 >= len(encoded) {
				return false, fmt.Errorf("toon: JSON normalization produced an unterminated escape")
			}
			if encoded[i+1] != 'u' {
				i += 2
				continue
			}
			codepoint, ok := jsonHex(encoded, i+2)
			if !ok {
				return false, fmt.Errorf("toon: JSON normalization produced an invalid Unicode escape")
			}
			if codepoint == 0xfffd {
				replacedInvalidUTF8 = true
			}
			if codepoint >= 0xdc00 && codepoint <= 0xdfff {
				return false, fmt.Errorf("toon: JSON normalization produced an unpaired surrogate")
			}
			if codepoint >= 0xd800 && codepoint <= 0xdbff {
				if i+12 > len(encoded) || encoded[i+6] != '\\' || encoded[i+7] != 'u' {
					return false, fmt.Errorf("toon: JSON normalization produced an unpaired surrogate")
				}
				low, ok := jsonHex(encoded, i+8)
				if !ok || low < 0xdc00 || low > 0xdfff {
					return false, fmt.Errorf("toon: JSON normalization produced an unpaired surrogate")
				}
				i += 12
				continue
			}
			i += 6
		}
		if i >= len(encoded) {
			return false, fmt.Errorf("toon: JSON normalization produced an unterminated string")
		}
	}
	return replacedInvalidUTF8, nil
}

func jsonHex(encoded []byte, start int) (uint16, bool) {
	if start+4 > len(encoded) {
		return 0, false
	}
	var value uint16
	for _, c := range encoded[start : start+4] {
		value <<= 4
		switch {
		case c >= '0' && c <= '9':
			value += uint16(c - '0')
		case c >= 'a' && c <= 'f':
			value += uint16(c-'a') + 10
		case c >= 'A' && c <= 'F':
			value += uint16(c-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}

func canonicalNumber(number string) string {
	negative := strings.HasPrefix(number, "-")
	if negative {
		number = number[1:]
	}

	coefficient := number
	exponent := new(big.Int)
	if at := strings.IndexAny(number, "eE"); at >= 0 {
		coefficient = number[:at]
		exponentText := strings.TrimPrefix(number[at+1:], "+")
		exponent.SetString(exponentText, 10)
	}

	fractionLength := 0
	if dot := strings.IndexByte(coefficient, '.'); dot >= 0 {
		fractionLength = len(coefficient) - dot - 1
		coefficient = coefficient[:dot] + coefficient[dot+1:]
	}
	digits := strings.TrimLeft(coefficient, "0")
	if digits == "" {
		return "0"
	}
	exponent.Sub(exponent, big.NewInt(int64(fractionLength)))
	for strings.HasSuffix(digits, "0") {
		digits = strings.TrimSuffix(digits, "0")
		exponent.Add(exponent, big.NewInt(1))
	}

	decimalPoint := new(big.Int).Add(exponent, big.NewInt(int64(len(digits))))
	minusFive := big.NewInt(-5)
	twentyOne := big.NewInt(21)
	var canonical string
	if decimalPoint.Cmp(minusFive) >= 0 && decimalPoint.Cmp(twentyOne) <= 0 {
		point := int(decimalPoint.Int64())
		switch {
		case point <= 0:
			canonical = "0." + strings.Repeat("0", -point) + digits
		case point >= len(digits):
			canonical = digits + strings.Repeat("0", point-len(digits))
		default:
			canonical = digits[:point] + "." + digits[point:]
		}
	} else {
		scientificExponent := new(big.Int).Sub(decimalPoint, big.NewInt(1))
		canonical = digits[:1]
		if len(digits) > 1 {
			canonical += "." + digits[1:]
		}
		canonical += "e"
		if scientificExponent.Sign() >= 0 {
			canonical += "+"
		}
		canonical += scientificExponent.String()
	}
	if negative {
		return "-" + canonical
	}
	return canonical
}
