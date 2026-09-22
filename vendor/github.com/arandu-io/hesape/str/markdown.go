package str

import (
	"regexp"
	"strconv"
	"strings"
)

// Markdown renders CommonMark, with the GitHub flavour's strikethrough, task
// lists, tables and bare-URL autolinking, as HTML.
//
//	Markdown("# Hello")   // "<h1>Hello</h1>\n"
//	Markdown("*hello*")   // "<p><em>hello</em></p>\n"
//
// Raw HTML passes through untouched. Rendering untrusted Markdown therefore
// renders untrusted HTML: sanitize it after this, before it reaches a page.
//
// There is nothing to configure and there will not be: the renderer is this
// file, and one way to spell a document is enough.
func Markdown(s string) string {
	var b strings.Builder
	renderBlocks(splitLines(s), &b)
	return b.String()
}

// InlineMarkdown renders only the inline part of CommonMark -- emphasis, code
// spans, links, images -- with no block element around it.
//
//	InlineMarkdown("**Hello World**") // "<strong>Hello World</strong>\n"
//
// A heading marker or a list marker is left standing as text, because there is
// no block parser to read it.
//
// The note on raw HTML in the comment on Markdown holds here too.
func InlineMarkdown(s string) string {
	return renderInline(strings.TrimRight(s, "\n")) + "\n"
}

// splitLines cuts the document into lines, taking either line ending and
// dropping the one at the end so that a trailing break makes no empty block.
func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

var (
	atxHeading     = regexp.MustCompile(`^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*#*[ \t]*$`)
	setextHeading  = regexp.MustCompile(`^ {0,3}(=+|-+)[ \t]*$`)
	thematicBreak  = regexp.MustCompile(`^ {0,3}((\*[ \t]*){3,}|(-[ \t]*){3,}|(_[ \t]*){3,})$`)
	fenceOpen      = regexp.MustCompile("^ {0,3}(`{3,}|~{3,})[ \t]*(.*)$")
	blockQuoteLine = regexp.MustCompile(`^ {0,3}>[ \t]?`)
	bulletItem     = regexp.MustCompile(`^( {0,3})([-*+])([ \t]+)(.*)$`)
	orderedItem    = regexp.MustCompile(`^( {0,3})(\d{1,9})([.)])([ \t]+)(.*)$`)
	taskMarker     = regexp.MustCompile(`^\[([ xX])\][ \t]+`)
	tableDelimiter = regexp.MustCompile(`^ {0,3}\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$`)
	// htmlBlockStart wants a name that is followed by what a tag is followed
	// by, so that an autolink such as <https://example.com> is not read as one.
	htmlBlockStart = regexp.MustCompile(`^ {0,3}<(/?[A-Za-z][A-Za-z0-9-]*(\s|/?>)|!--)`)
)

// renderBlocks reads block elements off the front of the lines until they run
// out, writing the HTML for each.
func renderBlocks(lines []string, b *strings.Builder) {
	for i := 0; i < len(lines); {
		line := lines[i]

		switch {
		case strings.TrimSpace(line) == "":
			i++

		case thematicBreak.MatchString(line):
			b.WriteString("<hr />\n")
			i++

		case atxHeading.MatchString(line):
			m := atxHeading.FindStringSubmatch(line)
			level := strconv.Itoa(len(m[1]))
			b.WriteString("<h" + level + ">" + renderInline(strings.TrimSpace(m[2])) + "</h" + level + ">\n")
			i++

		case fenceOpen.MatchString(line):
			i = renderFencedCode(lines, i, b)

		case blockQuoteLine.MatchString(line):
			i = renderBlockQuote(lines, i, b)

		case bulletItem.MatchString(line) || orderedItem.MatchString(line):
			i = renderList(lines, i, b)

		case isIndentedCode(line):
			i = renderIndentedCode(lines, i, b)

		case htmlBlockStart.MatchString(line):
			i = renderHTMLBlock(lines, i, b)

		default:
			i = renderParagraphOrTable(lines, i, b)
		}
	}
}

func isIndentedCode(line string) bool {
	return strings.HasPrefix(line, "    ") || strings.HasPrefix(line, "\t")
}

// renderFencedCode reads a fenced code block and writes it out with the info
// string as a language class.
func renderFencedCode(lines []string, i int, b *strings.Builder) int {
	m := fenceOpen.FindStringSubmatch(lines[i])
	fence, info := m[1], strings.TrimSpace(m[2])
	indent := len(lines[i]) - len(strings.TrimLeft(lines[i], " "))

	var body []string
	i++
	for i < len(lines) {
		trimmed := strings.TrimSpace(lines[i])
		if strings.HasPrefix(trimmed, fence[:1]) && len(trimmed) >= len(fence) && strings.Trim(trimmed, fence[:1]) == "" {
			i++
			break
		}
		body = append(body, trimLeadingSpaces(lines[i], indent))
		i++
	}

	b.WriteString("<pre><code")
	if info != "" {
		language, _, _ := strings.Cut(info, " ")
		b.WriteString(` class="language-` + escapeHTML(language) + `"`)
	}
	b.WriteString(">")
	for _, l := range body {
		b.WriteString(escapeHTML(l) + "\n")
	}
	b.WriteString("</code></pre>\n")
	return i
}

// renderIndentedCode reads the run of four-space-indented lines as a code
// block, keeping the blank lines that sit inside it.
func renderIndentedCode(lines []string, i int, b *strings.Builder) int {
	var body []string
	for i < len(lines) {
		if isIndentedCode(lines[i]) {
			body = append(body, trimLeadingSpaces(strings.Replace(lines[i], "\t", "    ", 1), 4))
			i++
			continue
		}
		if strings.TrimSpace(lines[i]) == "" && hasMoreIndentedCode(lines, i+1) {
			body = append(body, "")
			i++
			continue
		}
		break
	}
	b.WriteString("<pre><code>")
	for _, l := range body {
		b.WriteString(escapeHTML(l) + "\n")
	}
	b.WriteString("</code></pre>\n")
	return i
}

// hasMoreIndentedCode looks past a run of blank lines for another indented one,
// which is what keeps a blank line inside a code block instead of ending it.
func hasMoreIndentedCode(lines []string, i int) bool {
	for ; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) != "" {
			return isIndentedCode(lines[i])
		}
	}
	return false
}

// lastIndex is the index renderIndentedCode stopped at, which it signals by
// running the cursor off the end.
func lastIndex(lines []string, i int) int {
	if i > len(lines) {
		return len(lines)
	}
	return i
}

// renderBlockQuote reads the run of quoted lines, strips the markers and
// renders what is left as blocks of its own.
func renderBlockQuote(lines []string, i int, b *strings.Builder) int {
	var inner []string
	for i < len(lines) {
		switch {
		case blockQuoteLine.MatchString(lines[i]):
			inner = append(inner, blockQuoteLine.ReplaceAllString(lines[i], ""))
			i++
		case strings.TrimSpace(lines[i]) != "" && len(inner) > 0:
			// A lazy continuation line belongs to the paragraph inside the quote.
			inner = append(inner, lines[i])
			i++
		default:
			b.WriteString("<blockquote>\n")
			renderBlocks(inner, b)
			b.WriteString("</blockquote>\n")
			return i
		}
	}
	b.WriteString("<blockquote>\n")
	renderBlocks(inner, b)
	b.WriteString("</blockquote>\n")
	return i
}

// listItem is one item of a list, with the lines that belong to it and whether
// a blank line sat inside or in front of it.
type listItem struct {
	lines []string
	loose bool
}

// renderList reads a run of items sharing one marker type and writes the list.
func renderList(lines []string, i int, b *strings.Builder) int {
	ordered := orderedItem.MatchString(lines[i])
	start := ""
	if ordered {
		start = orderedItem.FindStringSubmatch(lines[i])[2]
	}
	marker := itemMarker(lines[i])

	var items []listItem
	loose := false
	for i < len(lines) {
		if strings.TrimSpace(lines[i]) == "" {
			// A blank line makes the list loose only if an item follows it.
			j := i
			for j < len(lines) && strings.TrimSpace(lines[j]) == "" {
				j++
			}
			if j < len(lines) && itemMarker(lines[j]) == marker {
				loose = true
				i = j
				continue
			}
			break
		}
		if itemMarker(lines[i]) != marker {
			break
		}

		content, indent := itemContent(lines[i])
		item := listItem{lines: []string{content}}
		i++
		for i < len(lines) {
			switch {
			case strings.TrimSpace(lines[i]) == "":
				if j := i + 1; j < len(lines) && countIndent(lines[j]) >= indent {
					item.lines = append(item.lines, "")
					item.loose = true
					i++
					continue
				}
			case countIndent(lines[i]) >= indent:
				item.lines = append(item.lines, trimLeadingSpaces(lines[i], indent))
				i++
				continue
			case itemMarker(lines[i]) == "" && !startsBlock(lines[i]):
				item.lines = append(item.lines, lines[i])
				i++
				continue
			}
			break
		}
		items = append(items, item)
		loose = loose || item.loose
	}

	tag := "ul"
	open := "<ul>"
	if ordered {
		tag = "ol"
		open = "<ol>"
		if start != "" && start != "1" {
			open = `<ol start="` + start + `">`
		}
	}

	b.WriteString(open + "\n")
	for _, item := range items {
		writeListItem(item, loose, b)
	}
	b.WriteString("</" + tag + ">\n")
	return i
}

// writeListItem writes one item, wrapping its text in a paragraph when the list
// is loose and leaving it bare when it is tight.
func writeListItem(item listItem, loose bool, b *strings.Builder) {
	body := strings.Join(item.lines, "\n")
	checkbox := ""
	if m := taskMarker.FindStringSubmatch(body); m != nil {
		checked := ""
		if m[1] != " " {
			checked = ` checked=""`
		}
		checkbox = `<input` + checked + ` disabled="" type="checkbox"> `
		body = body[len(m[0]):]
	}

	if !loose && !strings.Contains(strings.TrimSpace(body), "\n") {
		b.WriteString("<li>" + checkbox + renderInline(strings.TrimSpace(body)) + "</li>\n")
		return
	}
	b.WriteString("<li>\n")
	if checkbox != "" {
		b.WriteString(checkbox)
	}
	renderBlocks(splitLines(body), b)
	b.WriteString("</li>\n")
}

// itemMarker names the kind of list a line opens, so that a run of items with
// one marker is one list.
func itemMarker(line string) string {
	if m := bulletItem.FindStringSubmatch(line); m != nil {
		return "bullet" + m[2]
	}
	if m := orderedItem.FindStringSubmatch(line); m != nil {
		return "ordered" + m[3]
	}
	return ""
}

// itemContent is what an item line says and how far its continuation lines have
// to be indented to belong to it.
func itemContent(line string) (string, int) {
	if m := bulletItem.FindStringSubmatch(line); m != nil {
		return m[4], len(m[1]) + 1 + len(m[3])
	}
	m := orderedItem.FindStringSubmatch(line)
	return m[5], len(m[1]) + len(m[2]) + 1 + len(m[4])
}

// startsBlock reports whether a line opens a block of its own, which is what
// stops it from being read as the continuation of a paragraph.
func startsBlock(line string) bool {
	return thematicBreak.MatchString(line) || atxHeading.MatchString(line) ||
		fenceOpen.MatchString(line) || blockQuoteLine.MatchString(line) ||
		htmlBlockStart.MatchString(line)
}

// renderHTMLBlock passes a run of raw HTML through to the output untouched.
func renderHTMLBlock(lines []string, i int, b *strings.Builder) int {
	for i < len(lines) && strings.TrimSpace(lines[i]) != "" {
		b.WriteString(lines[i] + "\n")
		i++
	}
	return i
}

// renderParagraphOrTable reads the run of lines up to the next blank line or
// block start, and writes it as a table, a setext heading or a paragraph.
func renderParagraphOrTable(lines []string, i int, b *strings.Builder) int {
	var block []string
	for i < len(lines) {
		line := lines[i]
		if strings.TrimSpace(line) == "" {
			break
		}
		// The setext underline is read before the block starters, because a run
		// of dashes is also a thematic break and under a paragraph it is not.
		if len(block) > 0 && setextHeading.MatchString(line) {
			level := "2"
			if strings.HasPrefix(strings.TrimSpace(line), "=") {
				level = "1"
			}
			b.WriteString("<h" + level + ">" + renderInline(strings.TrimSpace(strings.Join(block, "\n"))) + "</h" + level + ">\n")
			return i + 1
		}
		if len(block) > 0 && (startsBlock(line) || itemMarker(line) != "") {
			break
		}
		block = append(block, line)
		i++
	}

	if len(block) >= 2 && strings.Contains(block[0], "|") && tableDelimiter.MatchString(block[1]) {
		writeTable(block, b)
		return i
	}
	b.WriteString("<p>" + renderInline(strings.TrimSpace(strings.Join(block, "\n"))) + "</p>\n")
	return i
}

// writeTable writes the GitHub table whose header and delimiter row have
// already been recognised.
func writeTable(block []string, b *strings.Builder) {
	alignments := tableAlignments(block[1])
	b.WriteString("<table>\n<thead>\n<tr>\n")
	for i, cell := range tableCells(block[0]) {
		b.WriteString("<th" + alignmentAt(alignments, i) + ">" + renderInline(cell) + "</th>\n")
	}
	b.WriteString("</tr>\n</thead>\n")
	if len(block) > 2 {
		b.WriteString("<tbody>\n")
		for _, row := range block[2:] {
			b.WriteString("<tr>\n")
			for i, cell := range tableCells(row) {
				b.WriteString("<td" + alignmentAt(alignments, i) + ">" + renderInline(cell) + "</td>\n")
			}
			b.WriteString("</tr>\n")
		}
		b.WriteString("</tbody>\n")
	}
	b.WriteString("</table>\n")
}

func tableCells(row string) []string {
	row = strings.TrimSpace(row)
	row = strings.TrimPrefix(row, "|")
	row = strings.TrimSuffix(row, "|")
	cells := strings.Split(row, "|")
	for i := range cells {
		cells[i] = strings.TrimSpace(cells[i])
	}
	return cells
}

func tableAlignments(delimiter string) []string {
	var out []string
	for _, cell := range tableCells(delimiter) {
		left, right := strings.HasPrefix(cell, ":"), strings.HasSuffix(cell, ":")
		switch {
		case left && right:
			out = append(out, "center")
		case right:
			out = append(out, "right")
		case left:
			out = append(out, "left")
		default:
			out = append(out, "")
		}
	}
	return out
}

func alignmentAt(alignments []string, i int) string {
	if i < len(alignments) && alignments[i] != "" {
		return ` align="` + alignments[i] + `"`
	}
	return ""
}

// trimLeadingSpaces removes up to n leading spaces, which is how a nested block
// is brought back to column zero before it is parsed again.
func trimLeadingSpaces(line string, n int) string {
	i := 0
	for i < n && i < len(line) && line[i] == ' ' {
		i++
	}
	return line[i:]
}

// countIndent is the number of leading spaces on a line, with a tab counting
// as four.
func countIndent(line string) int {
	n := 0
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case ' ':
			n++
		case '\t':
			n += 4
		default:
			return n
		}
	}
	return n
}

// htmlEscaper escapes the four characters that cannot stand for themselves in
// HTML text.
var htmlEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;")

func escapeHTML(s string) string { return htmlEscaper.Replace(s) }
