package routing

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// FormatRoutes renders the route table for the terminal, grouped by module and
// sorted by pattern.
//
// It is here, and not in the CLI, so that every consumer prints the same
// table: the CLI's route list and the route list on the error page are the
// same rows in the same order, and a developer comparing the two is comparing
// like with like.
//
// The name column is what a developer copies into Routes.Route instead of
// typing the path, so a route without one prints short rather than printing an
// empty column that looks like a value. A deprecated route prints its sunset
// date last, which is the date a reader plans against.
func FormatRoutes(routes []*Route) string {
	sorted := append([]*Route{}, routes...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Module != sorted[j].Module {
			return sorted[i].Module < sorted[j].Module
		}
		if sorted[i].Pattern != sorted[j].Pattern {
			return sorted[i].Pattern < sorted[j].Pattern
		}
		return sorted[i].Method < sorted[j].Method
	})

	var b strings.Builder
	module := ""
	for _, r := range sorted {
		if r.Module != module {
			module = r.Module
			fmt.Fprintf(&b, "\n%s\n", module)
		}
		fmt.Fprintf(&b, "  %s\n", routeRow(r))
	}
	if b.Len() == 0 {
		return "no routes registered\n"
	}
	return strings.TrimLeft(b.String(), "\n")
}

// routeColumnWidths are the widths of every column but the last one a row
// prints.
var routeColumnWidths = []int{7, 34, 24}

// routeRow renders one route as the columns it has: the method and the pattern
// always, the name when it has one, and the sunset date when it is deprecated.
//
// The last column of a row is never padded, so a row that stops early stops
// clean instead of trailing spaces into a column that holds nothing.
func routeRow(r *Route) string {
	cells := []string{r.Method, r.Pattern}
	if name := r.RouteName(); name != "" {
		cells = append(cells, name)
	}
	if _, sunset := r.GetDeprecation(); !sunset.IsZero() {
		cells = append(cells, "sunset "+sunset.UTC().Format(time.DateOnly))
	}

	var b strings.Builder
	for i, cell := range cells {
		if i > 0 {
			b.WriteString(" ")
		}
		if i == len(cells)-1 || i >= len(routeColumnWidths) {
			b.WriteString(cell)
			continue
		}
		fmt.Fprintf(&b, "%-*s", routeColumnWidths[i], cell)
	}
	return b.String()
}
