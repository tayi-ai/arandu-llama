//go:build kyse

package llama

@go
// IndexData is what a handler hands this page.
//
// A struct, never a map. A field that does not exist stops the build, and a
// name misspelled in the markup below is a compile error rather than a blank
// space on a page that answered 200.
type IndexData struct {
	// Heading is the line at the top of the page.
	Heading string
	// Records is the listing, one row each.
	Records []IndexRecord
}

// IndexRecord is one row of the listing.
type IndexRecord struct {
	// ID is the identifier the row is addressed by.
	ID string
	// Name is what the row shows.
	Name string
}
@endgo

<section class="mx-auto max-w-3xl px-6 py-12">
	<h1 class="text-3xl font-semibold tracking-tight">{{ .Heading }}</h1>

	@if(len(.Records) > 0)
		<ul class="mt-8 grid gap-3">
			@foreach(.Records as record)
				<li class="card p-4">
					<span class="text-sm font-semibold">{{ record.Name }}</span>
					<span class="text-muted-foreground ml-2 text-xs">{{ record.ID }}</span>
				</li>
			@endforeach
		</ul>
	@else
		<p class="text-muted-foreground mt-8 text-sm">Nothing here yet.</p>
	@endif
</section>
