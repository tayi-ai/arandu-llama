/* Arandu client behaviours.
 *
 * Everything interactive on a page is bound once, on `document`, and dispatched
 * by looking at `data-*` attributes on the element an event came from. Nothing
 * here reads an attribute and evaluates it: attributes carry data, never code.
 *
 * That is the whole reason this file exists rather than Alpine. Alpine compiles
 * every directive expression with `new AsyncFunction`, and this framework serves
 * `script-src 'self'` with no `unsafe-eval`, so every directive threw at the
 * point of evaluation and no client behaviour ever ran. Loosening the policy
 * would buy the behaviour back at the price of the policy, and building the same
 * evaluator by hand would reopen the same hole under a different name.
 *
 * Because it is delegation, markup that HTMX swaps in is live the moment it
 * lands: every behaviour this file ships needs no initialising and no tearing
 * down. The one sweep below stamps state that only the browser knows -- which
 * accent is in force -- onto attributes a stylesheet and a screen reader can
 * see. Skipping the sweep costs a checkmark, never an interaction.
 *
 * Those behaviours read no expression, keep no parallel state, and store
 * nothing per element: open, active and selected all live in the ARIA the
 * markup already has to carry, so the DOM is the state and there is only one
 * copy of it.
 *
 * # The one thing that does have a lifecycle
 *
 * An application registers behaviours of its own by name -- arandu.ui.define
 * and arandu.ui.action, below -- and those get mounted, updated and destroyed,
 * because a behaviour somebody else wrote may take a timer or an observer and
 * has to be told when its element is going away. That is the exception, it is
 * hooked to htmx's own events, and it changes nothing about the rule: the
 * attribute holds a name that is looked up in a map, never code that is run.
 *
 * Loading this file twice is a no-op.
 */
(function () {
	'use strict';

	var arandu = window.arandu = window.arandu || {};
	if (arandu.ui) return;
	arandu.ui = { version: 1 };

	var OPTION = '[role="option"]';
	var LINE = '[role="menuitem"]';
	var COPY_RESET = 1600;

	var copyTimers = new WeakMap();

	/* The element an event came from, or null when it was not an element. */
	function origin(event) {
		var node = event.target;
		return node && node.nodeType === 1 ? node : null;
	}

	/* Whether an item can be reached: not disabled, and not filtered away. */
	function usable(item) {
		return !!item &&
			item.getAttribute('aria-disabled') !== 'true' &&
			item.getAttribute('aria-hidden') !== 'true' &&
			!item.hasAttribute('disabled');
	}

	function list(container, selector) {
		if (!container) return [];
		var found = container.querySelectorAll(selector);
		var out = [];
		for (var i = 0; i < found.length; i++) {
			if (usable(found[i])) out.push(found[i]);
		}
		return out;
	}

	/* ---- active item -------------------------------------------------------
	 *
	 * Which item is active is held in one place, aria-activedescendant on the
	 * text box, because that is the attribute a screen reader reads. The class
	 * on the item is what the stylesheet paints, and it is kept in step here so
	 * the two can never disagree.
	 */

	function activeID(input) {
		return input ? input.getAttribute('aria-activedescendant') : null;
	}

	function setActive(input, item, container, selector, scroll) {
		if (container) {
			var all = container.querySelectorAll(selector);
			for (var i = 0; i < all.length; i++) {
				if (all[i] !== item) all[i].classList.remove('active');
			}
		}
		if (!item) {
			if (input) input.removeAttribute('aria-activedescendant');
			return;
		}
		item.classList.add('active');
		if (input) {
			if (item.id) input.setAttribute('aria-activedescendant', item.id);
			else input.removeAttribute('aria-activedescendant');
		}
		if (scroll && item.scrollIntoView) item.scrollIntoView({ block: 'nearest' });
	}

	function move(input, container, selector, step) {
		var all = list(container, selector);
		if (!all.length) return;

		var current = activeID(input);
		var at = -1;
		for (var i = 0; i < all.length; i++) {
			if (all[i].id && all[i].id === current) { at = i; break; }
		}
		var to = at < 0 ? (step > 0 ? 0 : all.length - 1) : (at + step + all.length) % all.length;
		setActive(input, all[to], container, selector, true);
	}

	/* ---- theme -------------------------------------------------------------
	 *
	 * The state itself is theme.js's: it applies the choice to <html> before the
	 * body is parsed and owns what is written to storage. This only asks it to
	 * change, and reflects the answer onto the buttons.
	 */

	function theme() {
		return arandu.theme || null;
	}

	function setMode(mode) {
		var store = theme();
		if (!mode || !store || typeof store.set !== 'function') return;
		store.set(mode);
		stamp(document);
	}

	/* ---- table ---------------------------------------------------------------
	 *
	 * The three things about a table only the browser can do: keep the count
	 * of chosen rows in step, switch a column off, and move a cell at a time
	 * when the table is a grid.
	 *
	 * Searching, ordering and paging are not among them, and that is the whole
	 * design. Each is an address: the header is a link, the search is a form,
	 * and htmx swaps what the server answers with. A copy of that written here
	 * would be a second way to order a table -- one that orders only the rows
	 * that were fetched, so page two of a list sorted by name holds whatever
	 * the server thought page two was.
	 *
	 * It was written once and taken out again. What it cost to keep correct
	 * -- a memo per row, a collator, a debounce, a cache thrown away on every
	 * sort and every hidden column -- is the tell: that is a table engine, and
	 * this framework already has one on the other side of the request.
	 *
	 * The sentence for the count comes from the server, in the same field the
	 * server rendered it from. Nothing here composes English.
	 */
	arandu.ui.define('table', {
		mounted: function (ctx) {
			var root = ctx.element;
			var props = ctx.props || {};

			/* ---- working a complete list in the browser --------------------
			 *
			 * Only when the page holds every row, which the server states with
			 * data-complete. Ordering a page of a longer list orders only what
			 * was sent, and filtering it hides matches that never arrived --
			 * both read as a table in the wrong order to everyone except
			 * whoever wrote it. When it is not complete none of this runs: the
			 * header is a link, the search is a form, and the server answers.
			 *
			 * The rows are moved, never rebuilt. Redrawing a body from strings
			 * loses every data attribute, every checked box, the focus and
			 * anything htmx bound to a row -- and turns text back into markup
			 * on the way. Moving a <tr> that is already in the document keeps
			 * all of it.
			 *
			 * Every sentence comes from the server, with its placeholders.
			 * Nothing here composes English.
			 */
			var complete = root.getAttribute('data-complete') === 'true';
			var body = function () { return root.querySelector('tbody'); };
			var allRows = function () {
				var tbody = body();
				if (!tbody) return [];
				return Array.prototype.filter.call(tbody.rows, function (row) {
					return !row.hasAttribute('data-empty-row');
				});
			};

			var say = function (name, slots) {
				var sentence = props[name] || '';
				for (var key in slots) {
					if (Object.prototype.hasOwnProperty.call(slots, key)) {
						sentence = sentence.split('{' + key + '}').join(slots[key]);
					}
				}
				return sentence;
			};

			/* One collator, made once. localeCompare builds one per call, and
			 * a sort is n log n calls -- measured over fifty thousand rows,
			 * eighty-eight milliseconds against fourteen. */
			var collator = null;
			var compare = function (x, y) {
				if (!collator) {
					collator = typeof Intl !== 'undefined' && Intl.Collator
						? new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })
						: { compare: function (a, b) { return a < b ? -1 : a > b ? 1 : 0; } };
				}
				return collator.compare(x, y);
			};

			/* The text a row is searched by, built once and kept on the row.
			 * It was built per row per keystroke; memoised it is four to eight
			 * times faster, and the gap widens with the table. It is dropped
			 * whenever a column is hidden, because a hidden column is not
			 * searched. */
			var haystack = function (row) {
				if (row.__hay === undefined) {
					var text = '';
					for (var i = 0; i < row.cells.length; i++) {
						if (row.cells[i].hidden) continue;
						text += (row.cells[i].textContent || '').toLowerCase() + ' ';
					}
					row.__hay = text;
				}
				return row.__hay;
			};
			var forget = function () {
				allRows().forEach(function (row) { row.__hay = undefined; });
			};

			var size = Number(props.pageSize) || 0;
			var page = 1;

			/* comparable is what a cell sorts by: the value the server handed
			 * over, or the text it draws. The two are different strings --
			 * "1.240,00" sorts before "860,00" as text and after it as money
			 * -- which is why the server is asked for the comparable form. */
			var comparable = function (row, at) {
				var cell = row.cells[at];
				if (!cell) return '';
				var raw = cell.getAttribute('data-sort-value');
				return raw === null ? (cell.textContent || '').trim() : raw;
			};

			/* Decorate, sort, undecorate: the key is read once per row instead
			 * of twice per comparison, and the column is numeric or it is not
			 * -- deciding per comparison lets one stray cell order the others
			 * by a rule that does not apply to them. */
			var order = function (at, dir) {
				var tbody = body();
				if (!tbody) return;
				var rows = allRows();
				var sign = dir === 'desc' ? -1 : 1;

				var keyed = new Array(rows.length);
				var numeric = false;
				var seenValue = false;
				for (var i = 0; i < rows.length; i++) {
					var raw = comparable(rows[i], at);
					var num = raw === '' ? NaN : Number(raw);
					/* Empty cells do not decide the kind of the column. One
					 * blank would otherwise send a column of money to the
					 * collator, where 1.5 sorts before 1.25 and -5 before -10. */
					if (raw !== '') {
						if (!seenValue) { numeric = true; seenValue = true; }
						if (isNaN(num)) numeric = false;
					}
					keyed[i] = { row: rows[i], raw: raw, num: num, blank: raw === '' };
				}
				/* Blanks go to one end and stay there whichever way the column
				 * is sorted: a row with nothing in the column is not the
				 * smallest value, it is the absence of one. */
				var by = numeric
					? function (a, b) { return (a.num - b.num) * sign; }
					: function (a, b) { return compare(a.raw, b.raw) * sign; };
				keyed.sort(function (a, b) {
					if (a.blank !== b.blank) return a.blank ? 1 : -1;
					if (a.blank) return 0;
					return by(a, b);
				});

				/* Through a fragment: appending each row to the live table is a
				 * layout per row, and the fragment makes it one. */
				var moved = document.createDocumentFragment();
				for (var m = 0; m < keyed.length; m++) moved.appendChild(keyed[m].row);
				tbody.appendChild(moved);
			};

			/* A search that matched nothing is not an empty table: the table
			 * has rows, and none of them is the answer. */
			var nothing = function (missing, columns) {
				var tbody = body();
				if (!tbody) return;
				var line = tbody.querySelector('[data-empty-row]');
				if (!missing) {
					if (line) line.remove();
					return;
				}
				if (line) return;
				var row = tbody.insertRow();
				row.setAttribute('data-empty-row', '');
				var cell = row.insertCell();
				cell.colSpan = columns;
				cell.className = 'data-table-nothing';
				cell.textContent = props.noResult || 'Nothing matched';
			};

			var showing = function (from, to, total) {
				var line = root.querySelector('[data-part="showing"]');
				if (!line) return;
				line.textContent = say('showing', { from: from, to: to, total: total });
			};

			/* The pager moves its window over the markup the server drew.
			 *
			 * Rebuilding it was the first thing tried and it is wrong: the
			 * <nav> loses its accessible name, the <a> become <button> and
			 * stop being addresses, every data-part and caller class goes, and
			 * the focus lands on <body> after every click. Moving the window
			 * keeps all of it -- the entries were already correct, they were
			 * pointing at the wrong numbers.
			 */
			var pager = function (pages) {
				var nav = root.querySelector('[data-part="pagination"]');
				if (!nav) return;
				nav.hidden = pages < 2;
				if (pages < 2) return;

				var numbered = Array.prototype.filter.call(
					nav.querySelectorAll('[data-part="link"]'),
					function (link) { return link.hasAttribute('data-page'); }
				);
				/* The window the server drew is as wide as it is; sliding it
				 * means renumbering the entries in place, not making more. */
				var width = numbered.length;
				var first = Math.max(1, Math.min(page - Math.floor(width / 2), pages - width + 1));

				numbered.forEach(function (link, at) {
					var n = first + at;
					var shown = n >= 1 && n <= pages;
					link.closest('li').hidden = !shown;
					if (!shown) return;
					link.textContent = String(n);
					link.setAttribute('data-page', String(n));
					link.setAttribute('aria-label', say('page', { n: n }));
					if (n === page) link.setAttribute('aria-current', 'page');
					else link.removeAttribute('aria-current');
				});

				var end = function (part, to, live) {
					var link = nav.querySelector('[data-part="' + part + '"]');
					if (!link) return;
					var item = link.closest('li');
					if (item) item.hidden = !live;
					link.setAttribute('data-page', String(to));
				};
				end('previous', Math.max(1, page - 1), page > 1);
				end('next', Math.min(pages, page + 1), page < pages);

				var gap = nav.querySelector('[data-part="ellipsis"]');
				if (gap && gap.closest('li')) gap.closest('li').hidden = first + width - 1 >= pages;
			};

			var show = function () {
				var query = '';
				var box = root.querySelector('[data-part="search"]');
				if (box) query = (box.value || '').trim().toLowerCase();

				var rows = allRows();
				var hits = new Array(rows.length);
				var matched = 0;
				for (var i = 0; i < rows.length; i++) {
					hits[i] = !query || haystack(rows[i]).indexOf(query) >= 0;
					if (hits[i]) matched++;
				}

				var pages = size > 0 ? Math.max(1, Math.ceil(matched / size)) : 1;
				if (page > pages) page = pages;
				var from = size > 0 ? (page - 1) * size : 0;
				var to = size > 0 ? page * size : Infinity;

				var seen = 0;
				for (var at = 0; at < rows.length; at++) {
					if (!hits[at]) { rows[at].hidden = true; continue; }
					rows[at].hidden = seen < from || seen >= to;
					seen++;
				}

				release();

				var head = root.querySelector('thead tr');
				nothing(matched === 0, head ? head.cells.length : 1);
				showing(matched === 0 ? 0 : from + 1, Math.min(from + (size || matched), matched), matched);
				pager(pages);
				count();
			};

			/* Only the rows on screen.
			 *
			 * A filtered-away row keeps its checkbox in the document, keeps its
			 * form attribute, and is still submitted -- so a header checkbox
			 * that ticked every box in the DOM would archive two hundred
			 * invoices from a table showing three. The count would say so, and
			 * nobody would read it in time.
			 *
			 * The server marks the same rows hidden for the same reason, and
			 * disables their boxes, so the hole is closed on both paths. */
			var boxes = function () {
				return Array.prototype.filter.call(
					root.querySelectorAll('[data-part="select"]'),
					function (box) {
						var row = box.closest('tr');
						return row && !row.hidden;
					}
				);
			};

			/* A row leaving the window takes its selection with it. Keeping it
			 * checked is a value the form would still send from a row nobody
			 * can see. */
			var release = function () {
				Array.prototype.forEach.call(
					root.querySelectorAll('[data-part="select"]'),
					function (box) {
						var row = box.closest('tr');
						if (!row) return;
						box.disabled = !!row.hidden;
						if (row.hidden) box.checked = false;
					}
				);
			};

			var count = function () {
				var line = root.querySelector('[data-part="count"]');
				if (!line) return;
				var all = boxes();
				var chosen = all.filter(function (box) { return box.checked; }).length;
				/* The template is the server's, written onto the element it
				 * renders into -- so the count is said in the language the page
				 * is served in and this file carries no sentence. */
				var template = line.getAttribute('data-selected-template') ||
					props.selected || '{n} of {total} rows selected';
				line.textContent = template
					.split('{n}').join(String(chosen))
					.split('{total}').join(String(all.length));

				all.forEach(function (box) {
					var row = box.closest('tr');
					if (!row) return;
					/* Written only where it differs: rewriting the attribute on
					 * every row on every keystroke is a mutation record per row
					 * for a value that did not change. */
					var now = box.checked ? 'true' : 'false';
					if (row.getAttribute('aria-selected') !== now) {
						row.setAttribute('aria-selected', now);
					}
				});

				var master = root.querySelector('[data-select-all]');
				if (master) {
					master.checked = all.length > 0 && chosen === all.length;
					/* Indeterminate is a property and not an attribute: there is
					 * no markup for "some", which is why the server cannot draw
					 * this state and this line exists. */
					master.indeterminate = chosen > 0 && chosen < all.length;
				}
			};

			ctx.change = function (event) {
				var master = event.target.closest('[data-select-all]');
				if (master) {
					boxes().forEach(function (box) { box.checked = master.checked; });
					count();
					return;
				}
				if (event.target.closest('[data-part="select"]')) { count(); return; }

				var toggle = event.target.closest('[data-column-toggle]');
				if (!toggle) return;
				var key = toggle.getAttribute('data-column-toggle');
				/* hidden rather than a class: a cell removed from the layout is
				 * also removed from the table's own column count, which is what
				 * keeps a screen reader reading the right header for the right
				 * cell. */
				each(root, '[data-column="' + key + '"]', function (cell) {
					cell.hidden = !toggle.checked;
				});
				/* A hidden column is not searched, so what every row is
				 * searched by has just changed. */
				if (complete) { forget(); show(); }
			};

			ctx.key = function (event) {
				if (root.getAttribute('data-navigable') !== 'true') return;
				var cell = event.target.closest('td, th');
				if (!cell || !root.contains(cell)) return;

				var row = cell.parentElement;
				var rows = Array.prototype.slice.call(root.querySelectorAll('tr'));
				var at = Array.prototype.indexOf.call(row.children, cell);
				var down = rows.indexOf(row);

				var go = function (r, c) {
					event.preventDefault();
					var target = rows[r] && rows[r].children[c];
					if (!target) return;
					rows.forEach(function (one) {
						Array.prototype.forEach.call(one.children, function (child) {
							child.setAttribute('tabindex', '-1');
						});
					});
					target.setAttribute('tabindex', '0');
					target.focus();
				};

				/* Read once, at the gesture: in a right-to-left document the
				 * arrow that points right moves towards the earlier column,
				 * and a grid that ignores that is a grid that moves backwards. */
				var rtl = getComputedStyle(root).direction === 'rtl';
				var forward = rtl ? -1 : 1;

				switch (event.key) {
					case 'ArrowRight': go(down, at + forward); break;
					case 'ArrowLeft': go(down, at - forward); break;
					case 'ArrowDown': go(down + 1, at); break;
					case 'ArrowUp': go(down - 1, at); break;
					case 'Home': go(down, 0); break;
					case 'End': go(down, row.children.length - 1); break;
				}
			};

			/* A navigable table needs one cell reachable by tab, and the server
			 * cannot know which -- it is wherever the person left off. The
			 * first is where they start. */
			if (root.getAttribute('data-navigable') === 'true') {
				each(root, 'td, th', function (cell) { cell.setAttribute('tabindex', '-1'); });
				var first = root.querySelector('th, td');
				if (first) first.setAttribute('tabindex', '0');
			}

			root.addEventListener('change', ctx.change);
			root.addEventListener('keydown', ctx.key);

			if (complete) {
				/* Every sortable header says which of the three states it is
				 * in. A header that only draws an arrow is a header a screen
				 * reader reads as a link. */
				var name = function (head) {
					var link = head.querySelector('[data-part="sort"]');
					if (!link) return;
					var state = head.getAttribute('aria-sort');
					var line = state === 'ascending' ? 'ascending'
						: state === 'descending' ? 'descending' : 'unsorted';
					link.setAttribute('aria-label',
						say(line, { column: (link.textContent || '').trim() }));
				};
				each(root, 'th[aria-sort]', name);

				/* The header stays a link. Taking the click before the browser
				 * follows it is what makes this an enhancement: with no script
				 * the same address loads the same order from the server. */
				ctx.sort = function (event) {
					var link = event.target.closest('[data-part="sort"]');
					if (!link) return;
					event.preventDefault();
					var head = link.closest('th');
					var at = Array.prototype.indexOf.call(head.parentElement.children, head);
					var dir = head.getAttribute('aria-sort') === 'ascending' ? 'desc' : 'asc';
					each(root, 'th[aria-sort]', function (one) { one.setAttribute('aria-sort', 'none'); });
					head.setAttribute('aria-sort', dir === 'desc' ? 'descending' : 'ascending');
					each(root, 'th[aria-sort]', name);
					order(at, dir);
					/* Reordering resets the page, as the server's own SortURL
					 * does -- otherwise the same gesture has two answers
					 * depending on where the rows are worked. */
					page = 1;
					show();
				};

				/* Debounced for the same reason the server search is: a word
				 * typed at speed is one answer and not eight. Shorter, because
				 * there is no network to wait for. */
				var pending = 0;
				ctx.filter = function (event) {
					if (!event.target.closest('[data-part="search"]')) return;
					window.clearTimeout(pending);
					pending = window.setTimeout(function () { page = 1; show(); }, 80);
				};

				ctx.turn = function (event) {
					var link = event.target.closest('[data-part="pagination"] [data-page]');
					if (!link) return;
					event.preventDefault();
					page = Number(link.getAttribute('data-page')) || 1;
					show();
					/* The entry moved under the pointer; the focus follows the
					 * number rather than the position, so a keyboard does not
					 * land on whatever now occupies that slot. */
					var again = root.querySelector('[data-part="pagination"] [data-page="' + page + '"]');
					if (again) again.focus();
				};

				ctx.resize = function (event) {
					var chooser = event.target.closest('[data-page-size]');
					if (!chooser) return;
					size = Number(chooser.value) || 0;
					page = 1;
					show();
				};

				root.addEventListener('click', ctx.sort);
				root.addEventListener('click', ctx.turn);
				root.addEventListener('input', ctx.filter);
				root.addEventListener('change', ctx.resize);
				ctx.refresh = function () { forget(); show(); };
				show();
			}

			count();
		},
		updated: function (ctx) {
			/* The body may be entirely different rows now. What each was
			 * searched by is the first thing that is no longer true. */
			if (ctx.refresh) ctx.refresh();
		},
		destroyed: function (ctx) {
			ctx.element.removeEventListener('change', ctx.change);
			ctx.element.removeEventListener('keydown', ctx.key);
			if (ctx.sort) ctx.element.removeEventListener('click', ctx.sort);
			if (ctx.turn) ctx.element.removeEventListener('click', ctx.turn);
			if (ctx.filter) ctx.element.removeEventListener('input', ctx.filter);
			if (ctx.resize) ctx.element.removeEventListener('change', ctx.resize);
		},
	});

	/* ---- dialog ---------------------------------------------------------------
	 *
	 * Opening and closing a <dialog> from an attribute, so nothing has to write
	 * an inline handler to do it.
	 *
	 * The components' own documentation used to say `onclick="ID.showModal()"`,
	 * and that is advice this project's policy refuses: the CSP is
	 * script-src 'self' with no 'unsafe-inline', which blocks an inline handler
	 * exactly as it blocks an inline <script>. The advice would work on a page
	 * that had loosened the policy, and on no page that had not -- which is the
	 * worst kind of wrong, because it works while somebody is building and
	 * fails when the headers go on.
	 *
	 * So it is a delegated attribute, like every other behaviour here: a name
	 * that is looked up, never a string that is evaluated.
	 *
	 *     <button data-dialog-open="confirm-delete">Delete</button>
	 *     <button data-dialog-close>Cancel</button>
	 *
	 * showModal rather than show, because a modal is what these panels are:
	 * the rest of the document goes inert, focus is trapped, and Escape closes
	 * -- all three from the element, none of them written here.
	 */
	document.addEventListener('click', function (event) {
		var open = event.target.closest('[data-dialog-open]');
		if (open) {
			var panel = document.getElementById(open.getAttribute('data-dialog-open'));
			/* A dialog that is not on the page is a page that changed under a
			 * button, and it is worth one line rather than a silent nothing. */
			if (!panel || typeof panel.showModal !== 'function') {
				miss('dialog', open.getAttribute('data-dialog-open'));
				return;
			}
			event.preventDefault();
			if (!panel.open) panel.showModal();
			return;
		}

		var close = event.target.closest('[data-dialog-close]');
		if (!close) return;
		/* The named dialog, or the one this button is inside -- so a close
		 * button needs no id when it sits in the panel it closes. */
		var named = close.getAttribute('data-dialog-close');
		var target = named ? document.getElementById(named) : close.closest('dialog');
		if (!target) return;
		event.preventDefault();
		target.close();
	});

	/* ---- calendar ------------------------------------------------------------
	 *
	 * A month grid: the arrow keys move a day at a time, the buttons move a
	 * month, and choosing writes the date into the input the calendar was
	 * pointed at.
	 *
	 * # The arithmetic is here and the words are not
	 *
	 * Moving to another month redraws the grid rather than asking the server,
	 * so a month change costs nothing. That is a second copy of the date
	 * arithmetic the component already did in Go, and it is worth saying so
	 * plainly rather than pretending otherwise -- what keeps the two from
	 * drifting is that neither decides anything a person reads. The month
	 * names, the column heads, the first day of the week and the bounds all
	 * arrive as props, drawn from the application's catalogue. Nothing here
	 * formats a name, reads a locale or picks a calendar system, so the grid
	 * this draws and the grid the server drew say the same words.
	 *
	 * # Why the cells are not buttons
	 *
	 * A month is forty-two cells. As buttons that is forty-two tab stops, and
	 * the grid role exists precisely so it is one: the arrows move within it,
	 * and tab leaves it. That is the contract a person already knows from
	 * every other calendar.
	 */
	arandu.ui.define('calendar', {
		mounted: function (ctx) {
			var root = ctx.element;
			var props = ctx.props || {};

			var pad = function (n) { return (n < 10 ? '0' : '') + n; };
			var iso = function (y, m, d) { return y + '-' + pad(m + 1) + '-' + pad(d); };

			var cells = function () {
				return Array.prototype.slice.call(root.querySelectorAll('[role="gridcell"]'));
			};
			var focusDate = function (date) {
				var all = cells();
				for (var at = 0; at < all.length; at++) {
					if (all[at].getAttribute('data-date') !== date) continue;
					all.forEach(function (one) { one.setAttribute('tabindex', '-1'); });
					all[at].setAttribute('tabindex', '0');
					all[at].focus();
					return true;
				}
				return false;
			};

			/* draw rewrites the grid for one month. It writes numbers, dates
			 * and state, and nothing else -- the elements are the ones the
			 * server drew, so every class and every part name is untouched. */
			var draw = function (year, month) {
				var first = new Date(Date.UTC(year, month, 1));
				var firstDay = Number(props.firstDay) || 0;
				var offset = (first.getUTCDay() - firstDay + 7) % 7;
				var start = new Date(Date.UTC(year, month, 1 - offset));

				var today = new Date();
				var todayISO = iso(today.getFullYear(), today.getMonth(), today.getDate());
				var chosen = root.getAttribute('data-value') || '';

				cells().forEach(function (cell, at) {
					var day = new Date(start.getTime());
					day.setUTCDate(day.getUTCDate() + at);
					var date = iso(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate());

					cell.textContent = String(day.getUTCDate());
					cell.setAttribute('data-date', date);
					attr(cell, 'data-outside', day.getUTCMonth() !== month ? 'true' : null);
					attr(cell, 'data-today', date === todayISO ? 'true' : null);
					cell.setAttribute('aria-selected', date === chosen ? 'true' : 'false');
					attr(cell, 'aria-disabled', outside(date) ? 'true' : null);
					cell.setAttribute('tabindex', '-1');
				});

				root.setAttribute('data-month', year + '-' + pad(month + 1));
				var title = root.querySelector('[data-part="title"]');
				var names = props.months || [];
				if (title && names.length === 12) title.textContent = names[month] + ' ' + year;
			};

			var attr = function (element, name, value) {
				if (value === null) element.removeAttribute(name);
				else element.setAttribute(name, value);
			};

			var outside = function (date) {
				if (props.min && date < props.min) return true;
				return !!(props.max && date > props.max);
			};

			var month = function () {
				var parts = (root.getAttribute('data-month') || '').split('-');
				return { year: Number(parts[0]), month: Number(parts[1]) - 1 };
			};

			var step = function (by) {
				var at = month();
				var moved = new Date(Date.UTC(at.year, at.month + by, 1));
				draw(moved.getUTCFullYear(), moved.getUTCMonth());
			};

			/* choose writes the day into the input the calendar points at, and
			 * fires the events a form and htmx are listening for -- setting
			 * .value fires neither on its own. */
			var choose = function (date) {
				root.setAttribute('data-value', date || '');
				cells().forEach(function (cell) {
					cell.setAttribute('aria-selected',
						date && cell.getAttribute('data-date') === date ? 'true' : 'false');
				});
				if (!props.target) return;
				var field = document.getElementById(props.target);
				if (!field) return;
				field.value = date || '';
				field.dispatchEvent(new Event('input', { bubbles: true }));
				field.dispatchEvent(new Event('change', { bubbles: true }));
			};

			ctx.click = function (event) {
				var stepper = event.target.closest('[data-calendar-step]');
				if (stepper) { step(Number(stepper.getAttribute('data-calendar-step')) || 1); return; }
				if (event.target.closest('[data-calendar-clear]')) { choose(''); return; }
				if (event.target.closest('[data-calendar-today]')) {
					var now = new Date();
					draw(now.getFullYear(), now.getMonth());
					var date = iso(now.getFullYear(), now.getMonth(), now.getDate());
					if (!outside(date)) { choose(date); focusDate(date); }
					return;
				}
				var cell = event.target.closest('[role="gridcell"]');
				if (cell && cell.getAttribute('aria-disabled') !== 'true') {
					choose(cell.getAttribute('data-date'));
					focusDate(cell.getAttribute('data-date'));
				}
			};

			ctx.key = function (event) {
				var cell = event.target.closest('[role="gridcell"]');
				if (!cell) return;
				var parts = cell.getAttribute('data-date').split('-');
				var day = new Date(Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2])));

				var move = function (days) {
					event.preventDefault();
					day.setUTCDate(day.getUTCDate() + days);
					var date = iso(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate());
					if (!focusDate(date)) {
						draw(day.getUTCFullYear(), day.getUTCMonth());
						focusDate(date);
					}
				};

				switch (event.key) {
					case 'ArrowRight': move(1); break;
					case 'ArrowLeft': move(-1); break;
					case 'ArrowDown': move(7); break;
					case 'ArrowUp': move(-7); break;
					/* Home and End are the ends of the week, which is the row
					 * the person is looking at -- not the ends of the month,
					 * which PageUp and PageDown reach. */
					case 'Home': move(-((day.getUTCDay() - (Number(props.firstDay) || 0) + 7) % 7)); break;
					case 'End': move(6 - ((day.getUTCDay() - (Number(props.firstDay) || 0) + 7) % 7)); break;
					case 'PageUp': event.preventDefault(); step(-1); break;
					case 'PageDown': event.preventDefault(); step(1); break;
					case 'Enter':
					case ' ':
						event.preventDefault();
						if (cell.getAttribute('aria-disabled') !== 'true') choose(cell.getAttribute('data-date'));
						break;
				}
			};

			root.addEventListener('click', ctx.click);
			root.addEventListener('keydown', ctx.key);
		},
		destroyed: function (ctx) {
			ctx.element.removeEventListener('click', ctx.click);
			ctx.element.removeEventListener('keydown', ctx.key);
		},
	});

	/* ---- roving ---------------------------------------------------------------
	 *
	 * One tab stop for a group of controls, with the arrow keys moving inside
	 * it. A toolbar, a menubar and a tree all want exactly this, so it is
	 * written once.
	 *
	 * The tab stop moves rather than multiplying: whichever control is current
	 * has tabindex 0 and every other has -1, so tabbing out and back in
	 * returns to where the person was rather than to the beginning.
	 */
	function roving(container, selector, options) {
		var settings = options || {};
		var items = function () {
			return Array.prototype.filter.call(
				container.querySelectorAll(selector),
				function (element) {
					return !element.disabled &&
						element.getAttribute('aria-disabled') !== 'true' &&
						element.offsetParent !== null;
				}
			);
		};

		var focus = function (element) {
			if (!element) return;
			items().forEach(function (one) { one.setAttribute('tabindex', '-1'); });
			element.setAttribute('tabindex', '0');
			element.focus();
		};

		container.addEventListener('keydown', function (event) {
			var all = items();
			var at = all.indexOf(document.activeElement);
			if (at < 0) return;

			var vertical = settings.vertical;
			var forward = vertical ? 'ArrowDown' : 'ArrowRight';
			var backward = vertical ? 'ArrowUp' : 'ArrowLeft';

			if (event.key === forward) {
				event.preventDefault();
				focus(all[(at + 1) % all.length]);
			} else if (event.key === backward) {
				event.preventDefault();
				focus(all[(at - 1 + all.length) % all.length]);
			} else if (event.key === 'Home') {
				event.preventDefault();
				focus(all[0]);
			} else if (event.key === 'End') {
				event.preventDefault();
				focus(all[all.length - 1]);
			} else if (settings.onKey) {
				settings.onKey(event, all, at, focus);
			}
		});

		/* A click also moves the tab stop, so the two ways of getting to a
		 * control agree about where the keyboard is. */
		container.addEventListener('click', function (event) {
			var hit = event.target.closest(selector);
			if (hit && items().indexOf(hit) >= 0) {
				items().forEach(function (one) { one.setAttribute('tabindex', '-1'); });
				hit.setAttribute('tabindex', '0');
			}
		});
	}

	/* ---- toolbar -------------------------------------------------------------
	 *
	 * A row of controls that is one tab stop. Nothing else: the controls do
	 * what they do, and this only decides which of them the keyboard is on.
	 */
	arandu.ui.define('toolbar', {
		mounted: function (ctx) {
			roving(ctx.element, '[data-part="control"]', {
				vertical: !!(ctx.props || {}).vertical,
			});
		},
	});

	/* ---- menubar -------------------------------------------------------------
	 *
	 * The row of menus along the top of an application.
	 *
	 * Moving sideways while a menu is open closes it and opens the next --
	 * that is the behaviour that makes a menubar quick, and it is the part
	 * most reimplementations leave out. Down opens and steps in; Escape closes
	 * and comes back to the trigger, which is where the person was.
	 */
	arandu.ui.define('menubar', {
		mounted: function (ctx) {
			var bar = ctx.element;

			var panelOf = function (trigger) {
				return document.getElementById(trigger.id.replace('-trigger-', '-panel-'));
			};
			var close = function (trigger) {
				var panel = panelOf(trigger);
				if (!panel) return;
				panel.setAttribute('aria-hidden', 'true');
				trigger.setAttribute('aria-expanded', 'false');
			};
			var closeAll = function () {
				bar.querySelectorAll('[data-part="trigger"]').forEach(close);
			};
			var open = function (trigger) {
				var panel = panelOf(trigger);
				if (!panel) return;
				closeAll();
				panel.setAttribute('aria-hidden', 'false');
				trigger.setAttribute('aria-expanded', 'true');
			};
			var isOpen = function (trigger) {
				return trigger.getAttribute('aria-expanded') === 'true';
			};

			roving(bar, '[data-part="trigger"]', {
				onKey: function (event, all, at) {
					if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') {
						event.preventDefault();
						open(all[at]);
						var first = panelOf(all[at]).querySelector('[role="menuitem"]:not([disabled])');
						if (first) first.focus();
					} else if (event.key === 'Escape') {
						closeAll();
					}
				},
			});

			/* The roving handler already moved the focus by the time this runs,
			 * so an open menu simply follows it to whatever is now focused. */
			bar.addEventListener('keyup', function (event) {
				if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
				var landed = document.activeElement;
				if (landed && landed.matches('[data-part="trigger"]') &&
					bar.querySelector('[data-part="trigger"][aria-expanded="true"]')) {
					open(landed);
				}
			});

			bar.addEventListener('click', function (event) {
				var trigger = event.target.closest('[data-part="trigger"]');
				if (!trigger) return;
				if (isOpen(trigger)) close(trigger); else open(trigger);
			});

			/* Inside a menu: up and down between entries, Escape back out to
			 * the trigger -- which is where the person came from and where
			 * they expect to be. */
			bar.addEventListener('keydown', function (event) {
				var entry = event.target.closest('[role="menuitem"]');
				if (!entry) return;
				var menu = entry.closest('[role="menu"]');
				var entries = Array.prototype.filter.call(
					menu.querySelectorAll('[role="menuitem"]'),
					function (one) { return !one.disabled; }
				);
				var at = entries.indexOf(entry);
				var trigger = document.getElementById(menu.getAttribute('aria-labelledby'));

				if (event.key === 'ArrowDown') {
					event.preventDefault();
					entries[(at + 1) % entries.length].focus();
				} else if (event.key === 'ArrowUp') {
					event.preventDefault();
					entries[(at - 1 + entries.length) % entries.length].focus();
				} else if (event.key === 'Escape') {
					event.preventDefault();
					close(trigger);
					trigger.focus();
				}
			});

			ctx.dismiss = function (event) {
				if (!bar.contains(event.target)) closeAll();
			};
			document.addEventListener('click', ctx.dismiss);
		},
		destroyed: function (ctx) {
			if (ctx.dismiss) document.removeEventListener('click', ctx.dismiss);
		},
	});

	/* ---- tree ----------------------------------------------------------------
	 *
	 * A hierarchy that is one tab stop.
	 *
	 * Right opens a closed branch and then steps into it; left closes an open
	 * one and then steps out to the parent. That two-stage behaviour is the
	 * whole tree contract and is what makes a deep tree navigable with two
	 * keys.
	 *
	 * Opening and closing is done here rather than fetched, because the rows
	 * are already in the markup -- a tree whose branches are fetched swaps its
	 * own rows and this behaviour re-mounts on what comes back.
	 */
	arandu.ui.define('tree', {
		mounted: function (ctx) {
			var tree = ctx.element;
			var multiple = !!(ctx.props || {}).multiple;

			var rows = function () {
				return Array.prototype.filter.call(
					tree.querySelectorAll('[role="treeitem"]'),
					function (row) { return row.getAttribute('aria-disabled') !== 'true'; }
				);
			};
			var levelOf = function (row) {
				return Number(row.getAttribute('aria-level') || '1');
			};
			var move = function (row) {
				if (!row) return;
				rows().forEach(function (one) { one.setAttribute('tabindex', '-1'); });
				row.setAttribute('tabindex', '0');
				row.focus();
			};
			var toggle = function (row, open) {
				if (!row.hasAttribute('aria-expanded')) return false;
				if (row.getAttribute('aria-expanded') === String(open)) return false;
				row.setAttribute('aria-expanded', String(open));
				/* The children are the following rows deeper than this one, up
				 * to the next row at the same level or shallower. */
				var all = Array.prototype.slice.call(tree.querySelectorAll('[role="treeitem"]'));
				var at = all.indexOf(row);
				var depth = levelOf(row);
				for (var next = at + 1; next < all.length; next++) {
					if (levelOf(all[next]) <= depth) break;
					all[next].hidden = !open || levelOf(all[next]) > depth + 1
						? !open || all[next].closest('[aria-expanded="false"]') !== null
						: !open;
				}
				return true;
			};

			tree.addEventListener('keydown', function (event) {
				var row = event.target.closest('[role="treeitem"]');
				if (!row) return;
				var all = rows();
				var at = all.indexOf(row);

				if (event.key === 'ArrowDown') {
					event.preventDefault();
					move(all[Math.min(at + 1, all.length - 1)]);
				} else if (event.key === 'ArrowUp') {
					event.preventDefault();
					move(all[Math.max(at - 1, 0)]);
				} else if (event.key === 'ArrowRight') {
					event.preventDefault();
					if (!toggle(row, true) && at + 1 < all.length &&
						levelOf(all[at + 1]) > levelOf(row)) {
						move(all[at + 1]);
					}
				} else if (event.key === 'ArrowLeft') {
					event.preventDefault();
					if (!toggle(row, false)) {
						for (var back = at - 1; back >= 0; back--) {
							if (levelOf(all[back]) < levelOf(row)) { move(all[back]); break; }
						}
					}
				} else if (event.key === 'Home') {
					event.preventDefault();
					move(all[0]);
				} else if (event.key === 'End') {
					event.preventDefault();
					move(all[all.length - 1]);
				} else if (event.key === 'Enter' || event.key === ' ') {
					if (!multiple) {
						all.forEach(function (one) { one.removeAttribute('aria-selected'); });
					}
					row.setAttribute('aria-selected',
						row.getAttribute('aria-selected') === 'true' && multiple ? 'false' : 'true');
				}
			});

			tree.addEventListener('click', function (event) {
				var toggleHit = event.target.closest('[data-part="toggle"]');
				var row = event.target.closest('[role="treeitem"]');
				if (!row) return;
				move(row);
				if (toggleHit) {
					toggle(row, row.getAttribute('aria-expanded') !== 'true');
				}
			});
		},
	});

	/* ---- carousel ------------------------------------------------------------
	 *
	 * The two buttons beside a scroll-snapping row.
	 *
	 * The scrolling itself belongs to the browser -- a drag, a swipe and the
	 * scrollbar all already work -- so this only asks it to scroll by one
	 * slide. That is why a carousel with no script still works and this only
	 * adds a way to do it with a pointer that is not dragging.
	 */
	arandu.ui.define('carousel', {
		mounted: function (ctx) {
			var root = ctx.element;
			var viewport = root.querySelector('[data-part="viewport"]');
			if (!viewport) return;
			var vertical = !!(ctx.props || {}).vertical;

			ctx.step = function (event) {
				var button = event.target.closest('[data-carousel-step]');
				if (!button) return;
				var direction = Number(button.getAttribute('data-carousel-step')) || 1;
				var slide = viewport.querySelector('[data-part="slide"]');
				var by = slide
					? (vertical ? slide.offsetHeight : slide.offsetWidth)
					: (vertical ? viewport.clientHeight : viewport.clientWidth);
				var to = {};
				to[vertical ? 'top' : 'left'] = by * direction;
				to.behavior = 'smooth';
				viewport.scrollBy(to);
			};
			root.addEventListener('click', ctx.step);
		},
		destroyed: function (ctx) {
			if (ctx.step) ctx.element.removeEventListener('click', ctx.step);
		},
	});

	/* ---- file-upload ---------------------------------------------------------
	 *
	 * Dropping files into the real file input.
	 *
	 * The input is what holds them, always: a drop writes into input.files
	 * through a DataTransfer, so the form submits exactly as if the files had
	 * been picked. Nothing here keeps a list of its own, which is what would
	 * otherwise disagree with the field the moment somebody picks again.
	 *
	 * dragover has to be cancelled or the browser opens the file instead, and
	 * that is the one line without which nothing works.
	 */
	arandu.ui.define('file-upload', {
		mounted: function (ctx) {
			var root = ctx.element;
			var zone = root.querySelector('[data-dropzone]');
			var input = root.querySelector('[data-part="input"]');
			if (!zone || !input) return;

			var stop = function (event) {
				event.preventDefault();
				event.stopPropagation();
			};

			ctx.over = function (event) {
				stop(event);
				zone.setAttribute('data-dragging', 'true');
			};
			ctx.leave = function (event) {
				stop(event);
				zone.removeAttribute('data-dragging');
			};
			ctx.drop = function (event) {
				stop(event);
				zone.removeAttribute('data-dragging');
				if (!event.dataTransfer || !event.dataTransfer.files.length) return;

				var carried = new DataTransfer();
				var files = event.dataTransfer.files;
				var many = input.multiple;
				for (var at = 0; at < files.length; at++) {
					carried.items.add(files[at]);
					if (!many) break;
				}
				input.files = carried.files;
				/* change, not input: it is the event a file input fires when
				 * files are picked, and it is what the hx-trigger listens for. */
				input.dispatchEvent(new Event('change', { bubbles: true }));
			};

			zone.addEventListener('dragenter', ctx.over);
			zone.addEventListener('dragover', ctx.over);
			zone.addEventListener('dragleave', ctx.leave);
			zone.addEventListener('drop', ctx.drop);
		},
		destroyed: function (ctx) {
			var zone = ctx.element.querySelector('[data-dropzone]');
			if (!zone) return;
			zone.removeEventListener('dragenter', ctx.over);
			zone.removeEventListener('dragover', ctx.over);
			zone.removeEventListener('dragleave', ctx.leave);
			zone.removeEventListener('drop', ctx.drop);
		},
	});

	/* ---- optimistic-toggle ---------------------------------------------------
	 *
	 * Flip now, reconcile after.
	 *
	 * The flip happens on the press and the request goes out behind it. What
	 * the server answers with is this control redrawn in the state it actually
	 * stored, which replaces the optimistic one -- so a disagreement corrects
	 * itself without anything here comparing the two.
	 *
	 * A failed request is the case that needs handling: nothing comes back to
	 * replace the control, so the flip is undone here and the count with it.
	 */
	arandu.ui.define('optimistic-toggle', {
		mounted: function (ctx) {
			var button = ctx.element;

			var paint = function (pressed) {
				button.setAttribute('aria-pressed', pressed ? 'true' : 'false');
				var on = button.querySelector('[data-toggle-on]');
				var off = button.querySelector('[data-toggle-off]');
				if (on) on.hidden = !pressed;
				if (off) off.hidden = pressed;
			};

			ctx.flip = function () {
				ctx.was = button.getAttribute('aria-pressed') === 'true';
				var now = !ctx.was;
				paint(now);

				var count = button.querySelector('[data-toggle-count]');
				if (count) {
					var value = parseInt(count.textContent, 10);
					if (!isNaN(value)) {
						ctx.count = count.textContent;
						count.textContent = String(Math.max(0, value + (now ? 1 : -1)));
					}
				}
			};

			ctx.revert = function () {
				if (ctx.was === undefined) return;
				paint(ctx.was);
				var count = button.querySelector('[data-toggle-count]');
				if (count && ctx.count !== undefined) count.textContent = ctx.count;
			};

			button.addEventListener('click', ctx.flip);
			/* htmx:responseError covers a 4xx or 5xx; htmx:sendError covers the
			 * request never arriving. Both leave the control as it was drawn,
			 * and both have to put it back. */
			button.addEventListener('htmx:responseError', ctx.revert);
			button.addEventListener('htmx:sendError', ctx.revert);
		},
		destroyed: function (ctx) {
			ctx.element.removeEventListener('click', ctx.flip);
			ctx.element.removeEventListener('htmx:responseError', ctx.revert);
			ctx.element.removeEventListener('htmx:sendError', ctx.revert);
		},
	});

	/* ---- number-input --------------------------------------------------------
	 *
	 * The two buttons beside a number box.
	 *
	 * The stepping itself is the browser's -- stepUp and stepDown honour min,
	 * max and step, and clamp rather than overshoot -- so this file decides
	 * nothing about arithmetic. What it adds is the press, and the input event
	 * afterwards, because stepUp does not fire one and everything watching the
	 * box for changes would otherwise never hear about these.
	 *
	 * The buttons are aria-hidden and out of the tab order in the markup: a
	 * keyboard already steps this box with the arrow keys, and two more tab
	 * stops per number on a form is a form nobody can get through.
	 */
	arandu.ui.define('number-input', {
		mounted: function (ctx) {
			var root = ctx.element;
			var input = root.querySelector('[data-part="input"]');
			if (!input) return;

			ctx.step = function (event) {
				var button = event.target.closest('[data-number-step]');
				if (!button || button.disabled || input.disabled || input.readOnly) return;
				var direction = Number(button.getAttribute('data-number-step'));
				if (!direction) return;
				try {
					if (direction > 0) input.stepUp(); else input.stepDown();
				} catch (error) {
					/* stepUp throws on a box whose value is not a number the
					 * element can read -- a half-typed exponent, an empty box in
					 * some engines. Starting from zero is the answer a person
					 * pressing "up" on an empty box expects. */
					input.value = String(direction > 0 ? 1 : -1);
				}
				input.dispatchEvent(new Event('input', { bubbles: true }));
				input.dispatchEvent(new Event('change', { bubbles: true }));
			};
			root.addEventListener('click', ctx.step);
		},
		destroyed: function (ctx) {
			if (ctx.step) ctx.element.removeEventListener('click', ctx.step);
		},
	});

	/* ---- copy --------------------------------------------------------------
	 *
	 * The text is read out of an attribute and handed to the clipboard as text.
	 * Which of the two words the button reads is the stylesheet's, keyed on the
	 * marker set here.
	 */

	function copy(button) {
		var text = button.getAttribute('data-copy-text');
		if (text === null || !navigator.clipboard) return;

		navigator.clipboard.writeText(text).then(function () {
			window.clearTimeout(copyTimers.get(button));
			button.setAttribute('data-copied', '');
			copyTimers.set(button, window.setTimeout(function () {
				button.removeAttribute('data-copied');
				copyTimers.delete(button);
			}, COPY_RESET));
		}, function () {
			/* No clipboard permission, or an insecure origin. The line is on the
			 * page and can be selected; saying nothing beats saying "copied" to
			 * somebody whose clipboard is empty. */
		});
	}

	/* ---- combobox ----------------------------------------------------------
	 *
	 * The list comes from the server on every keystroke, over HTMX, and is
	 * swapped into the listbox whole. Nothing here filters and nothing here
	 * fetches: this opens and closes the popover, walks the options that are
	 * present, and writes the chosen one into the two inputs.
	 */

	function comboboxParts(box) {
		return {
			input: box.querySelector('input[role="combobox"]'),
			listbox: box.querySelector('[role="listbox"]'),
			popover: box.querySelector('[data-popover]'),
			value: box.querySelector('[data-combobox-value]')
		};
	}

	function setComboboxOpen(box, open) {
		var parts = comboboxParts(box);
		if (parts.input) parts.input.setAttribute('aria-expanded', open ? 'true' : 'false');
		if (parts.popover) parts.popover.setAttribute('aria-hidden', open ? 'false' : 'true');
		if (!open) setActive(parts.input, null, parts.listbox, OPTION, false);
	}

	function comboboxOpen(box) {
		var input = box.querySelector('input[role="combobox"]');
		return !!input && input.getAttribute('aria-expanded') === 'true';
	}

	function chooseOption(box, option) {
		if (!usable(option)) return;
		var parts = comboboxParts(box);

		if (parts.listbox) {
			var all = parts.listbox.querySelectorAll(OPTION);
			for (var i = 0; i < all.length; i++) {
				if (all[i] === option) all[i].setAttribute('aria-selected', 'true');
				else all[i].removeAttribute('aria-selected');
			}
		}
		if (parts.value) parts.value.value = option.getAttribute('data-value') || '';
		if (parts.input) {
			var label = option.getAttribute('data-label');
			parts.input.value = label === null ? option.textContent.trim() : label;
		}
		setComboboxOpen(box, false);
	}

	function comboboxKey(event, box, input) {
		var parts = comboboxParts(box);

		if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
			event.preventDefault();
			setComboboxOpen(box, true);
			move(input, parts.listbox, OPTION, event.key === 'ArrowDown' ? 1 : -1);
			return;
		}
		if (event.key === 'Enter') {
			if (input.getAttribute('aria-expanded') !== 'true') return;
			event.preventDefault();
			var id = activeID(input);
			var option = id ? document.getElementById(id) : null;
			if (option && parts.listbox && parts.listbox.contains(option)) chooseOption(box, option);
			return;
		}
		if (event.key === 'Escape') setComboboxOpen(box, false);
	}

	/* ---- command palette ---------------------------------------------------
	 *
	 * Every line is in the document from the first paint and stays there. What
	 * is typed sets aria-hidden on the lines it does not match, and the
	 * stylesheet does the rest: it hides a hidden line, hides a group whose
	 * lines all went, and draws the empty message when none are left. Nothing is
	 * fetched, nothing is removed, and the search box's own value is the query
	 * -- there is no second copy of it to fall out of step.
	 */

	function commandParts(palette) {
		return {
			input: palette.querySelector('input[role="combobox"]'),
			menu: palette.querySelector('[role="menu"]')
		};
	}

	function filterCommand(palette) {
		var parts = commandParts(palette);
		if (!parts.menu) return;

		var query = parts.input ? parts.input.value.trim().toLowerCase() : '';
		var lines = parts.menu.querySelectorAll(LINE);
		for (var i = 0; i < lines.length; i++) {
			var line = lines[i];
			var haystack = (line.getAttribute('data-search') || line.textContent || '').toLowerCase();
			if (query && haystack.indexOf(query) === -1) line.setAttribute('aria-hidden', 'true');
			else line.removeAttribute('aria-hidden');
		}

		/* A line that was active and has just been filtered away is a line the
		 * Enter key would still follow. */
		var current = activeID(parts.input);
		var active = current ? document.getElementById(current) : null;
		if (current && (!active || !parts.menu.contains(active) || !usable(active))) {
			setActive(parts.input, null, parts.menu, LINE, false);
		}
	}

	function commandKey(event, palette, input) {
		var parts = commandParts(palette);

		if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
			event.preventDefault();
			move(input, parts.menu, LINE, event.key === 'ArrowDown' ? 1 : -1);
			return;
		}
		if (event.key === 'Enter') {
			event.preventDefault();
			var id = activeID(input);
			var line = id ? document.getElementById(id) : null;
			if (line && parts.menu && parts.menu.contains(line) && usable(line)) line.click();
			return;
		}
		if (event.key === 'Escape') {
			input.value = '';
			filterCommand(palette);
			setActive(input, null, parts.menu, LINE, false);
		}
	}

	/* ---- range slider ------------------------------------------------------
	 *
	 * The server draws the filled track and the number, so a slider is right
	 * before this runs and stays right if it never does. This keeps both in step
	 * with a thumb that is moving.
	 */

	function paintSlider(track) {
		var min = Number(track.min);
		var max = Number(track.max);
		var value = Number(track.value);
		if (!isFinite(min)) min = 0;
		if (!isFinite(max)) max = 100;
		if (!isFinite(value)) value = min;

		var span = max - min;
		var fill = span > 0 ? ((value - min) / span) * 100 : 0;
		track.style.setProperty('--slider-value', fill + '%');

		var field = track.closest('[data-slider]');
		var output = field ? field.querySelector('[data-slider-output]') : null;
		if (output) output.textContent = track.value;
	}

	/* ---- the sweep ---------------------------------------------------------
	 *
	 * Two things a server cannot write and delegation cannot reach, because they
	 * are state rather than events. Which accent is in force lives in the
	 * browser, so no cached page can carry aria-current on the right button; and
	 * the vendored stylesheet paints a hover highlight of its own until a
	 * component says it is driving one, which is what the initialised marker
	 * says. Both are idempotent, and neither gates any behaviour.
	 */

	function each(scope, selector, fn) {
		if (scope.nodeType === 1 && scope.matches(selector)) fn(scope);
		var found = scope.querySelectorAll(selector);
		for (var i = 0; i < found.length; i++) fn(found[i]);
	}

	function stamp(scope) {
		var node = scope && (scope.nodeType === 1 || scope.nodeType === 9) ? scope : document;
		var mode = document.documentElement.getAttribute('data-theme') || 'auto';

		each(node, '[data-theme-mode]', function (button) {
			button.setAttribute('aria-checked', button.getAttribute('data-theme-mode') === mode ? 'true' : 'false');
		});
		each(node, '[data-combobox]', function (box) {
			box.setAttribute('data-combobox-initialized', 'true');
		});
		each(node, '[data-command]', function (palette) {
			palette.setAttribute('data-command-initialized', 'true');
		});
	}

	/* ---- the registry ------------------------------------------------------
	 *
	 * Everything above is a behaviour this file ships. This is how an
	 * application adds one of its own without an inline handler.
	 *
	 * An application serves a script of its own -- registered with
	 * view.RegisterAsset, from the origin, like every other asset -- and calls:
	 *
	 *     arandu.ui.action('archive-message', function (event, element) { ... });
	 *
	 *     arandu.ui.define('message-actions', {
	 *         mounted:   function (ctx) { ... },
	 *         updated:   function (ctx) { ... },
	 *         destroyed: function (ctx) { ... },
	 *     });
	 *
	 * The markup names them and carries nothing else:
	 *
	 *     <div data-kyse-behavior="message-actions" data-kyse-props='{"confirm":true}'>
	 *       <button data-kyse-on-click="archive-message">Archive</button>
	 *
	 * # Why a name and not the code
	 *
	 * The attribute holds a key into a map. Nothing here parses it, compiles it
	 * or evaluates it, so the policy stays script-src 'self' with no
	 * unsafe-eval and this file keeps the property that made it exist instead of
	 * Alpine. A name that is not registered does nothing and says so once in the
	 * console -- which is a page missing a behaviour, never a page running one
	 * somebody typed into a form.
	 *
	 * It is the same shape as every delegated attribute above -- data-copy,
	 * data-dialog-open, data-slider: a catalogue rather than a parser is what
	 * keeps an attribute data instead of code.
	 */

	var actions = {};
	var behaviours = {};

	arandu.ui.action = function (name, fn) {
		if (typeof name !== 'string' || typeof fn !== 'function') return;
		actions[name] = fn;
	};

	arandu.ui.define = function (name, hooks) {
		if (typeof name !== 'string' || !hooks) return;
		behaviours[name] = hooks;
		/* Registration can arrive after the markup: a deferred application
		 * script runs once, and by then the document is parsed. Mounting what
		 * is already on the page is what makes the order not matter. */
		mount(document);
	};

	/* named answers a lookup and reports a miss once per name.
	 *
	 * Once, because the alternative is a console line per element per swap on a
	 * page whose behaviour is misspelled -- which buries the first one, and the
	 * first one is the whole message.
	 *
	 * And not before the page has loaded. This file and the application's script
	 * are both deferred, so the first sweep runs before a single define() has;
	 * warning there reported every behaviour on the page as unregistered, on
	 * every load, moments before registering all of them. What is left after
	 * window load is a real miss, and the sweep below says so. */
	var reported = {};
	var loaded = false;

	function named(map, kind, name) {
		if (Object.prototype.hasOwnProperty.call(map, name)) return map[name];
		if (loaded) miss(kind, name);
		return null;
	}

	function miss(kind, name) {
		if (reported[kind + ':' + name]) return;
		reported[kind + ':' + name] = true;
		if (window.console && console.warn) {
			console.warn('arandu.ui: no ' + kind + ' is registered as "' + name + '"');
		}
	}

	/* context is what a hook receives: the element, and the props the server
	 * wrote beside it.
	 *
	 * The props are parsed once and kept on the element, so updated and
	 * destroyed see the same object mounted did -- a hook that stored something
	 * on ctx.props finds it there later, which is the only place per-element
	 * state can live without this file keeping a second copy of the DOM. */
	function context(element) {
		if (!element.__kyse) {
			var props = {};
			var raw = element.getAttribute('data-kyse-props');
			if (raw) {
				try {
					props = JSON.parse(raw);
				} catch (e) {
					/* The server encodes this, so a parse failure is a bug here
					 * rather than something a visitor did. The behaviour still
					 * mounts, with no props, because half a page is worse. */
					if (window.console && console.warn) {
						console.warn('arandu.ui: the props of "' + element.getAttribute('data-kyse-behavior') + '" are not JSON');
					}
				}
			}
			element.__kyse = { element: element, props: props };
		}
		return element.__kyse;
	}

	/* justMounted holds what mounted since the last settle finished.
	 *
	 * htmx fires htmx:load from a settle task and htmx:afterSettle right after
	 * the tasks run -- `se(l.tasks,…);se(l.elts,…afterSettle…)` in the bundle --
	 * so without this every element that arrived in a swap got mounted and then
	 * immediately updated. Two hooks that always fire together are one hook, and
	 * a behaviour that did its setting up in mounted did it twice. */
	var justMounted = new WeakSet();

	function mount(scope) {
		var node = scope && (scope.nodeType === 1 || scope.nodeType === 9) ? scope : document;
		each(node, '[data-kyse-behavior]', function (element) {
			if (element.getAttribute('data-kyse-mounted') === 'true') return;
			var hooks = named(behaviours, 'behaviour', element.getAttribute('data-kyse-behavior'));
			if (!hooks) return;
			element.setAttribute('data-kyse-mounted', 'true');
			justMounted.add(element);
			if (typeof hooks.mounted === 'function') hooks.mounted(context(element));
		});
	}

	function update(scope) {
		var node = scope && (scope.nodeType === 1 || scope.nodeType === 9) ? scope : document;
		each(node, '[data-kyse-behavior][data-kyse-mounted="true"]', function (element) {
			if (justMounted.has(element)) {
				justMounted.delete(element);
				return;
			}
			var hooks = behaviours[element.getAttribute('data-kyse-behavior')];
			if (hooks && typeof hooks.updated === 'function') hooks.updated(context(element));
		});
	}

	function destroy(element) {
		if (!element || element.nodeType !== 1) return;
		each(element, '[data-kyse-behavior][data-kyse-mounted="true"]', function (mounted) {
			var hooks = behaviours[mounted.getAttribute('data-kyse-behavior')];
			if (hooks && typeof hooks.destroyed === 'function') hooks.destroyed(context(mounted));
			mounted.removeAttribute('data-kyse-mounted');
			mounted.__kyse = null;
		});
	}

	/* dispatch runs the action named for this event, if there is one.
	 *
	 * The attribute is data-kyse-on-<event>, so one lookup per delegated
	 * listener covers every action for that event on the page. */
	function dispatch(event, type) {
		var from = origin(event);
		if (!from) return;
		var element = from.closest('[data-kyse-on-' + type + ']');
		if (!element) return;
		var fn = named(actions, 'action', element.getAttribute('data-kyse-on-' + type));
		if (fn) fn(event, element);
	}

	/* ---- delegation --------------------------------------------------------
	 *
	 * Six listeners, all on document, all reading attributes rather than
	 * running them. Change and submit carry no behaviour of this file's own and
	 * exist for the registry: a form is submitted and a select is changed, and
	 * neither reaches a click listener.
	 */

	document.addEventListener('change', function (event) { dispatch(event, 'change'); });
	document.addEventListener('submit', function (event) { dispatch(event, 'submit'); });

	document.addEventListener('click', function (event) {
		dispatch(event, 'click');

		var from = origin(event);
		if (!from) return;

		var copyButton = from.closest('[data-copy]');
		if (copyButton) copy(copyButton);

		var mode = from.closest('[data-theme-mode]');
		if (mode) setMode(mode.getAttribute('data-theme-mode'));

		var option = from.closest('[data-combobox] ' + OPTION);
		if (option) chooseOption(option.closest('[data-combobox]'), option);

		/* Clicking away closes: the outside click every combobox on the page
		 * cares about, decided by containment rather than by a listener each. */
		var boxes = document.querySelectorAll('[data-combobox]');
		for (var i = 0; i < boxes.length; i++) {
			if (!boxes[i].contains(from) && comboboxOpen(boxes[i])) setComboboxOpen(boxes[i], false);
		}

		var input = from.closest('[data-combobox] input[role="combobox"]');
		if (input) setComboboxOpen(input.closest('[data-combobox]'), true);
	});

	document.addEventListener('input', function (event) {
		dispatch(event, 'input');

		var from = origin(event);
		if (!from) return;

		var track = from.closest('[data-slider-track]');
		if (track) paintSlider(track);

		var boxInput = from.closest('[data-combobox] input[role="combobox"]');
		if (boxInput) {
			var box = boxInput.closest('[data-combobox]');
			var parts = comboboxParts(box);
			setComboboxOpen(box, true);
			setActive(boxInput, null, parts.listbox, OPTION, false);
		}

		var paletteInput = from.closest('[data-command] input[role="combobox"]');
		if (paletteInput) filterCommand(paletteInput.closest('[data-command]'));
	});

	document.addEventListener('keydown', function (event) {
		dispatch(event, 'keydown');

		var from = origin(event);
		if (!from || !from.matches('input[role="combobox"]')) return;

		var box = from.closest('[data-combobox]');
		if (box) { comboboxKey(event, box, from); return; }

		var palette = from.closest('[data-command]');
		if (palette) commandKey(event, palette, from);
	});

	/* The mouse moves the active item so that the keyboard and the pointer
	 * never highlight two different rows. */
	document.addEventListener('pointermove', function (event) {
		var from = origin(event);
		if (!from) return;

		var option = from.closest('[data-combobox] ' + OPTION);
		if (option) {
			if (!usable(option)) return;
			var box = option.closest('[data-combobox]');
			var parts = comboboxParts(box);
			if (activeID(parts.input) !== option.id) {
				setActive(parts.input, option, parts.listbox, OPTION, false);
			}
			return;
		}

		var line = from.closest('[data-command] ' + LINE);
		if (line) {
			if (!usable(line)) return;
			var palette = line.closest('[data-command]');
			var pieces = commandParts(palette);
			if (activeID(pieces.input) !== line.id) {
				setActive(pieces.input, line, pieces.menu, LINE, false);
			}
		}
	});

	/* ---------------------------------------------------------------------
	 * Motion, on the platform's own animation engine.
	 *
	 * CSS covers almost everything a page needs: a hover, a transition, and --
	 * with animation-timeline: view() -- an element that arrives as it is
	 * scrolled to. Two things it does not cover well are a sequence whose steps
	 * are offset from one another, and an animation that has to run backwards on
	 * the way out. Those are what this is for, and it is why the list of what it
	 * can do is short: an effect CSS already expresses belongs in the stylesheet,
	 * where it survives this file failing to load.
	 *
	 * It is Element.animate -- the Web Animations API, which every current
	 * browser ships. That matters beyond taste: the animation libraries a page
	 * would otherwise reach for are built on this same call, and what they add is
	 * a nicer way to write it. Writing it directly costs the sugar and saves the
	 * dependency, the bytes, and the policy exception a third-party script would
	 * need under script-src 'self'.
	 *
	 * It reads no expression, like the rest of this file. `data-stagger` names an
	 * effect from a closed list, and `data-stagger-step` and
	 * `data-stagger-duration` are numbers clamped to a sane range; anything else
	 * is ignored. A catalogue rather than a parser is what keeps an attribute
	 * data instead of code.
	 *
	 * Nothing here is required for a page to be usable. Every element is at its
	 * final state before this runs and is put back to it if anything goes wrong,
	 * so a browser without IntersectionObserver, a reader who asked for less
	 * motion, and this file failing to parse all produce the same page: the one
	 * with everything already in place.
	 */
	var EFFECTS = {
		rise: [
			{ opacity: 0, transform: 'translateY(20px)' },
			{ opacity: 1, transform: 'none' }
		],
		fade: [
			{ opacity: 0 },
			{ opacity: 1 }
		],
		/* Used for a row of cards: each one arrives a little scaled down and a
		 * little low, so it settles into place rather than blinking on. The scale
		 * reads as depth and the small rise carries the eye; on their own each was
		 * too slight to see, which is the whole of the "did it animate?" report. */
		settle: [
			{ opacity: 0, transform: 'scale(.94) translateY(10px)' },
			{ opacity: 1, transform: 'none' }
		]
	};

	var STAGGER_MS = 70;

	/* Long enough to read as motion. At 460ms with the slight transforms above,
	 * the animation was over before the eye found it -- a reader would ask whether
	 * anything moved. The distance grew and the duration with it; the ease still
	 * starts fast and settles, so the row does not feel slow, it feels placed. */
	var DURATION_MS = 640;
	var EASE = 'cubic-bezier(.16,1,.3,1)';

	/* Whether motion is wanted at all.
	 *
	 * Asked at the moment of use rather than once at load, because a reader can
	 * change the setting without reloading the page, and the honest answer is the
	 * one the system gives now. */
	function motionWanted() {
		return !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	}

	/* Runs one container's children, offset from one another.
	 *
	 * The container is unobserved before the first frame rather than after the
	 * last: this runs once by design, and an element that scrolled out and back
	 * in mid-animation would otherwise restart from nothing while the reader was
	 * looking at it. */
	function runStagger(container) {
		var name = container.getAttribute('data-stagger');
		var frames = EFFECTS[name] || EFFECTS.rise;

		var step = parseInt(container.getAttribute('data-stagger-step'), 10);
		if (!(step >= 0 && step <= 400)) step = STAGGER_MS;

		/* A grid whose cards want a slower, more deliberate arrival than the row
		 * default sets data-stagger-duration; out of range or absent, the default
		 * stands. The component catalogue is the first caller: its cards read as
		 * hurried at the shared 640ms. */
		var duration = parseInt(container.getAttribute('data-stagger-duration'), 10);
		if (!(duration >= 200 && duration <= 2000)) duration = DURATION_MS;

		var children = container.children;
		for (var i = 0; i < children.length; i++) {
			children[i].animate(frames, {
				duration: duration,
				delay: i * step,
				easing: EASE,
				/* No fill. The element is already at its final state in the
				 * document, so the animation plays and hands it back rather
				 * than holding it: an animation that ends is an element the
				 * browser stops compositing. */
				fill: 'none'
			});
		}
	}

	function watchStagger(root) {
		if (!motionWanted()) return;
		if (typeof IntersectionObserver !== 'function') return;
		if (!Element.prototype.animate) return;

		var containers = root.querySelectorAll('[data-stagger]:not([data-stagger-done])');
		if (!containers.length) return;

		var seen = new IntersectionObserver(function (entries) {
			for (var i = 0; i < entries.length; i++) {
				if (!entries[i].isIntersecting) continue;
				var container = entries[i].target;
				seen.unobserve(container);
				container.setAttribute('data-stagger-done', '');
				runStagger(container);
			}
		}, { rootMargin: '0px 0px -12% 0px' });

		for (var j = 0; j < containers.length; j++) seen.observe(containers[j]);
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', function () { stamp(document); watchStagger(document); mount(document); });
	} else {
		stamp(document);
		watchStagger(document);
		mount(document);
	}

	/* htmx:load fires for markup that has just been inserted, which is where a
	 * behaviour on it is mounted. The two sweeps beside it were always
	 * idempotent; mount is too, by the marker it writes.
	 *
	 * afterSettle fires once the swap has settled, on markup that may have been
	 * there before -- so it is updated and never mounted, and an element that
	 * arrived in this same swap has already had mounted called by the line
	 * above rather than updated, which is the distinction the two hooks are
	 * for.
	 *
	 * beforeCleanupElement is the one hook this file has that is not a sweep:
	 * htmx calls it with an element it is about to remove, which is the last
	 * moment a behaviour can give back a timer, an observer or a listener it
	 * took. Without it the only cost is a leak, which is the kind that is
	 * invisible until a page has been open for an hour. */
	document.addEventListener('htmx:load', function (event) {
		stamp(event.target);
		watchStagger(event.target);
		mount(event.target);
	});
	document.addEventListener('htmx:afterSettle', function (event) { update(event.target); });
	document.addEventListener('htmx:beforeCleanupElement', function (event) { destroy(event.target); });

	/* The one-time code, typed one square at a time.
	 *
	 * The squares are the visible inputs and none of them submits; a hidden
	 * input beside them carries the whole code under the field's name, kept in
	 * step on every keystroke. So the server reads one field and never has to
	 * join six of them.
	 *
	 * # What it does about the small things
	 *
	 * Typing moves to the next square, and backspace on an empty one moves back
	 * -- otherwise fixing a mistake means clicking. Arrow keys walk the row. A
	 * code pasted into any square fills all of them, because that is what
	 * copying six digits out of a message means, and a code longer than the
	 * field is cut rather than refused.
	 *
	 * # It does not decide
	 *
	 * Whether the code is the right one is the server's answer, and nothing here
	 * asks. data-code-complete says every square is filled, which is a count.
	 */
	arandu.ui.define('one-time-code', {
		mounted: function (ctx) {
			var root = ctx.element;
			var squares = Array.prototype.slice.call(root.querySelectorAll('[data-part="square"]'));
			var raw = root.querySelector('[data-part="raw"]');
			if (!squares.length) return;

			var accepts = ctx.props.alphanumeric ? /[0-9a-zA-Z]/ : /[0-9]/;

			function collect() {
				var code = squares.map(function (one) { return one.value; }).join('');
				if (raw) raw.value = code;
				root.setAttribute('data-code-complete', code.length === squares.length ? 'true' : 'false');
			}

			function spread(text, from) {
				var kept = String(text).split('').filter(function (one) { return accepts.test(one); });
				for (var i = 0; i < kept.length && from + i < squares.length; i++) {
					squares[from + i].value = kept[i];
				}
				var landed = Math.min(from + kept.length, squares.length - 1);
				squares[landed].focus();
				collect();
			}

			ctx.otp = { onInput: [], onKeyDown: [], onPaste: [], onFocus: [] };

			squares.forEach(function (square, at) {
				var onInput = function () {
					var typed = square.value;
					if (typed.length > 1) {
						/* More than one character in a box that takes one is a
						 * paste the browser routed here as input. */
						square.value = '';
						spread(typed, at);
						return;
					}
					if (typed && !accepts.test(typed)) {
						square.value = '';
						collect();
						return;
					}
					if (typed && at + 1 < squares.length) {
						squares[at + 1].focus();
					}
					collect();
				};

				var onKeyDown = function (event) {
					if (event.key === 'Backspace' && !square.value && at > 0) {
						/* Backspace on an empty square goes back and clears the
						 * one it lands on: otherwise fixing a mistake means
						 * pressing it twice. */
						event.preventDefault();
						squares[at - 1].value = '';
						squares[at - 1].focus();
						collect();
						return;
					}
					if (event.key === 'ArrowLeft' && at > 0) {
						event.preventDefault();
						squares[at - 1].focus();
					}
					if (event.key === 'ArrowRight' && at + 1 < squares.length) {
						event.preventDefault();
						squares[at + 1].focus();
					}
				};

				var onPaste = function (event) {
					event.preventDefault();
					var text = (event.clipboardData || window.clipboardData).getData('text');
					spread(text, at);
				};

				/* Selecting what is in the square means the next keystroke
				 * replaces it, which is what somebody correcting one digit
				 * expects. */
				var onFocus = function () { square.select(); };

				square.addEventListener('input', onInput);
				square.addEventListener('keydown', onKeyDown);
				square.addEventListener('paste', onPaste);
				square.addEventListener('focus', onFocus);
				ctx.otp.onInput.push(onInput);
				ctx.otp.onKeyDown.push(onKeyDown);
				ctx.otp.onPaste.push(onPaste);
				ctx.otp.onFocus.push(onFocus);
			});

			collect();
		},
		destroyed: function (ctx) {
			if (!ctx.otp) return;
			var squares = Array.prototype.slice.call(
				ctx.element.querySelectorAll('[data-part="square"]'));
			squares.forEach(function (square, at) {
				square.removeEventListener('input', ctx.otp.onInput[at]);
				square.removeEventListener('keydown', ctx.otp.onKeyDown[at]);
				square.removeEventListener('paste', ctx.otp.onPaste[at]);
				square.removeEventListener('focus', ctx.otp.onFocus[at]);
			});
		},
	});

	/* The mask, which formats a field as somebody types in it.
	 *
	 * The pattern is a string of tokens, and the vocabulary is the one anybody
	 * who has written a mask before already knows:
	 *
	 *     0  a digit, required
	 *     9  a digit, optional
	 *     #  a digit, repeated to the end
	 *     A  a letter or a digit
	 *     S  a letter
	 *
	 * Everything else in the pattern is written through as it stands.
	 *
	 * # What the form sends is the raw value
	 *
	 * The visible field shows 123.456.789-00 and is not the field that submits.
	 * Beside it is a hidden input carrying 12345678900, kept in step on every
	 * keystroke, and that is what has the form name. So nothing on the server
	 * has to know a mask existed, and no query stores punctuation because
	 * somebody forgot to strip it in one of the places that read the field.
	 *
	 * The component writes both, so a caller cannot get half of it.
	 *
	 * # Alternatives
	 *
	 * A field that takes either of two shapes is given both, and the shortest
	 * one that still holds what has been typed is the one applied. A phone
	 * number grows into its longer form at the digit that needs it rather than
	 * jumping there.
	 *
	 * # It is not validation
	 *
	 * A mask says what may be typed. Whether what was typed means anything is
	 * the server's answer, and this asks nothing of it -- the same division the
	 * password behaviour above keeps.
	 */
	arandu.ui.define('mask', {
		mounted: function (ctx) {
			var input = ctx.element.matches('input') ? ctx.element : ctx.element.querySelector('input:not([type="hidden"])');
			if (!input) return;

			var patterns = maskPatterns(ctx.props);
			if (!patterns.length) return;

			var raw = ctx.element.querySelector('[data-part="raw"]');

			var reverse = ctx.props.reverse === true;

			ctx.format = function () {
				var pattern = maskFor(patterns, input.value);
				var before = input.value;
				var caret = input.selectionStart;
				var formatted = maskApply(pattern, before, reverse);

				if (formatted !== before) {
					input.value = formatted;
					if (reverse) {
						/* Filling from the right means the caret belongs at the
						 * end: every keystroke pushes what came before it left. */
						input.setSelectionRange(formatted.length, formatted.length);
					} else {
						/* The caret moves by however many characters the mask
						 * inserted before it, so typing in the middle of a value
						 * does not throw the cursor to the end. */
						var typed = maskUnmask(pattern, before.slice(0, caret)).length;
						var at = maskCaretFor(pattern, typed);
						input.setSelectionRange(at, at);
					}
				}
				if (raw) raw.value = maskUnmask(pattern, input.value);

				/* The field says whether the pattern is filled, so a stylesheet
				 * can show it and nothing has to ask this file. It is not a
				 * verdict: the server decides whether the value means anything,
				 * and a complete shape is only a shape. */
				input.setAttribute('data-mask-complete',
					maskComplete(pattern, input.value) ? 'true' : 'false');
			};

			input.addEventListener('input', ctx.format);
			/* Emptying a field that never matched, on the way out, is what the
			 * plugin this borrows its vocabulary from calls clearIfNotMatch. A
			 * half-typed value left behind is a value somebody submits without
			 * looking. */
			if (ctx.props.clearIfNotMatch === true) {
				input.addEventListener('blur', function () {
					var pattern = maskFor(patterns, input.value);
					if (input.value && !maskComplete(pattern, input.value)) {
						input.value = '';
						if (raw) raw.value = '';
						input.setAttribute('data-mask-complete', 'false');
					}
				});
			}
			/* A paste arrives as an input event in every browser this ships to,
			 * so there is nothing extra to listen for -- and a value the server
			 * rendered is formatted once, here, rather than left raw until
			 * somebody types. */
			ctx.format();
		},
		destroyed: function (ctx) {
			var input = ctx.element.matches('input') ? ctx.element : ctx.element.querySelector('input:not([type="hidden"])');
			if (input && ctx.format) input.removeEventListener('input', ctx.format);
		},
	});

	/* maskTokens is the dictionary, and it is closed: a mask needing a sixth
	 * token would be a mask nobody can read without this table beside them. */
	var maskTokens = {
		'0': { accepts: /[0-9]/ },
		'9': { accepts: /[0-9]/, optional: true },
		'#': { accepts: /[0-9]/, repeating: true },
		'A': { accepts: /[0-9a-zA-Z]/ },
		'S': { accepts: /[a-zA-Z]/ },
	};

	/* maskPatterns reads the pattern or patterns out of the props. */
	function maskPatterns(props) {
		var declared = (props || {}).pattern;
		if (typeof declared === 'string') return declared ? [declared] : [];
		if (Object.prototype.toString.call(declared) === '[object Array]') {
			return declared.filter(function (one) { return typeof one === 'string' && one; });
		}
		return [];
	}

	/* maskCapacity is how many characters a pattern's tokens accept, and -1 for
	 * one that repeats and therefore has no end. */
	function maskCapacity(pattern) {
		var n = 0;
		for (var i = 0; i < pattern.length; i++) {
			var token = maskTokens[pattern.charAt(i)];
			if (!token) continue;
			if (token.repeating) return -1;
			n++;
		}
		return n;
	}

	/* maskFor picks the shortest pattern that still holds what was typed. */
	function maskFor(patterns, value) {
		if (patterns.length === 1) return patterns[0];

		var widest = patterns[0];
		for (var i = 0; i < patterns.length; i++) {
			if (maskCapacity(patterns[i]) > maskCapacity(widest)) widest = patterns[i];
		}
		var typed = maskUnmask(widest, value).length;

		var best = null;
		for (var j = 0; j < patterns.length; j++) {
			var size = maskCapacity(patterns[j]);
			if (size < 0 || size < typed) continue;
			if (best === null || size < maskCapacity(best)) best = patterns[j];
		}
		return best || widest;
	}

	/* maskApply formats a value, dropping what the pattern cannot accept.
	 *
	 * Dropped rather than refused: what somebody pastes is usually the
	 * formatted value, and refusing its punctuation would mean refusing a paste
	 * of exactly what this produces. */
	function maskApply(pattern, value, reverse) {
		if (reverse) return maskApplyReverse(pattern, value);
		var out = '';
		var at = 0;
		for (var cursor = 0; cursor < pattern.length && at < value.length;) {
			var symbol = pattern.charAt(cursor);
			var token = maskTokens[symbol];
			if (!token) {
				out += symbol;
				if (value.charAt(at) === symbol) at++;
				cursor++;
				continue;
			}
			if (!token.accepts.test(value.charAt(at))) { at++; continue; }
			out += value.charAt(at);
			at++;
			if (!token.repeating) cursor++;
		}
		/* A separator with nothing behind it is punctuation somebody is about to
		 * type past, and showing it moves the caret for no reason. */
		while (out.length && !/[0-9a-zA-Z]/.test(out.charAt(out.length - 1))) {
			out = out.slice(0, -1);
		}
		return out;
	}

	/* maskApplyReverse fills a pattern from the right.
	 *
	 * It is how money is typed: the first digit somebody enters is the last one
	 * of the value, and every one after pushes the rest left past the
	 * separators. Filling from the left instead would put the first keystroke
	 * in the thousands and read 1 as 1.000,00.
	 *
	 * A separator with nothing in front of it is dropped for the same reason a
	 * trailing one is: it is punctuation the next keystroke is about to fill. */
	function maskApplyReverse(pattern, value) {
		var out = '';
		var at = value.length - 1;
		for (var cursor = pattern.length - 1; cursor >= 0 && at >= 0;) {
			var symbol = pattern.charAt(cursor);
			var token = maskTokens[symbol];
			if (!token) {
				out = symbol + out;
				if (value.charAt(at) === symbol) at--;
				cursor--;
				continue;
			}
			if (!token.accepts.test(value.charAt(at))) { at--; continue; }
			out = value.charAt(at) + out;
			at--;
			if (!token.repeating) cursor--;
		}
		while (out.length && !/[0-9a-zA-Z]/.test(out.charAt(0))) {
			out = out.slice(1);
		}
		return out;
	}

	/* maskUnmask gives back what the form sends: the characters the tokens
	 * accept, with everything the pattern writes through removed. */
	function maskUnmask(pattern, value) {
		var out = '';
		var at = 0;
		for (var cursor = 0; cursor < pattern.length && at < value.length;) {
			var symbol = pattern.charAt(cursor);
			var token = maskTokens[symbol];
			if (!token) {
				if (value.charAt(at) === symbol) at++;
				cursor++;
				continue;
			}
			if (!token.accepts.test(value.charAt(at))) { at++; continue; }
			out += value.charAt(at);
			at++;
			if (!token.repeating) cursor++;
		}
		return out;
	}

	/* maskComplete reports whether every required token is filled.
	 *
	 * Required, so a pattern whose tail is optional is complete without it:
	 * 0009 is complete at three digits. It answers about the shape and nothing
	 * else -- eleven digits are eleven digits, and whether they mean anything is
	 * a question with arithmetic in it that belongs to whoever owns the
	 * document. */
	function maskComplete(pattern, value) {
		var required = 0;
		for (var i = 0; i < pattern.length; i++) {
			var token = maskTokens[pattern.charAt(i)];
			if (!token || token.optional) continue;
			required++;
			if (token.repeating) break;
		}
		return maskUnmask(pattern, value).length >= required;
	}

	/* maskCaretFor is where the caret sits after a given number of accepted
	 * characters: past the separators that precede them. */
	function maskCaretFor(pattern, typed) {
		if (typed <= 0) return 0;
		var seen = 0;
		for (var i = 0; i < pattern.length; i++) {
			var token = maskTokens[pattern.charAt(i)];
			if (token) {
				seen++;
				if (seen >= typed) return i + 1;
			}
		}
		return pattern.length;
	}

	/* The password box, which is the one behaviour this file owns.
	 *
	 * Every other behaviour is the application's: this file registers two by way
	 * of example and nothing else. This one is different because the component
	 * that emits it ships in kyse, so a project that draws a password field
	 * writes data-kyse-behavior="password" without having asked for it -- and
	 * until now nothing registered that name. The panel never opened, the
	 * checklist never ticked, and the console said the behaviour was missing on
	 * every page with a sign-up form on it. A component that emits a name its
	 * own ecosystem does not answer is a component with a hole in it, and the
	 * hole belongs here rather than in every project.
	 *
	 * The props are the policy's AppliedRules verbatim -- min, max, letters,
	 * mixedCase, numbers, symbols, uncompromised -- keyed exactly as the
	 * requirement lines are, so a line is matched to its rule by name and never
	 * by position or by its wording.
	 *
	 * # What it does not do
	 *
	 * It does not decide. The server validates the password when the form is
	 * submitted, with the same policy, and nothing here is consulted for that:
	 * a checklist is a courtesy to whoever is typing, and treating it as the
	 * rule would put the decision in the browser. uncompromised is the plainest
	 * case -- only the server can answer it -- so its line never ticks here.
	 */
	arandu.ui.define('password', {
		mounted: function (ctx) {
			var root = ctx.element;
			var input = root.querySelector('[data-part="input"]');
			if (!input) return;

			var reveal = root.querySelector('[data-part="reveal"]');
			if (reveal) {
				reveal.addEventListener('click', function () {
					var hidden = input.type === 'password';
					input.type = hidden ? 'text' : 'password';
					reveal.setAttribute('aria-pressed', hidden ? 'true' : 'false');
					/* The two names are in the markup, so this file carries no
					 * sentence and translates nothing. The control is an icon,
					 * so the name is the only thing a screen reader has: it has
					 * to say which state pressing lands in, and it flips with
					 * the picture rather than after it. */
					var name = reveal.getAttribute(hidden ? 'data-reveal-shown' : 'data-reveal-hidden');
					if (name) reveal.setAttribute('aria-label', name);
					toggle(reveal.querySelector('[data-reveal-icon-shown]'), hidden);
					toggle(reveal.querySelector('[data-reveal-icon-hidden]'), !hidden);
					/* A label written as text rather than as a picture is still
					 * supported: a caller who put the words inside the button
					 * gets them swapped the same way. */
					toggle(reveal.querySelector('[data-reveal-shown]'), hidden);
					toggle(reveal.querySelector('[data-reveal-hidden]'), !hidden);
				});
			}

			ctx.judge = function () { judge(root, input.value, ctx.props || {}); };
			input.addEventListener('input', ctx.judge);
			ctx.judge();
		},
		destroyed: function (ctx) {
			var input = ctx.element.querySelector('[data-part="input"]');
			if (input && ctx.judge) input.removeEventListener('input', ctx.judge);
		},
	});

	/* toggle shows or hides one of the reveal labels. */
	function toggle(element, shown) {
		if (element) element.hidden = !shown;
	}

	/* judge ticks the lines a password already satisfies and moves the meter.
	 *
	 * The meter is the count of satisfied requirements over the number of them,
	 * and not a score: a number nobody can derive from what is on the screen is
	 * a number nobody can act on, and "why is it orange" has to have an answer
	 * the checklist gives.
	 */
	function judge(root, value, rules) {
		var met = {
			min: typeof rules.min !== 'number' || value.length >= rules.min,
			max: typeof rules.max !== 'number' || value.length <= rules.max,
			letters: !rules.letters || /\p{L}/u.test(value),
			mixedCase: !rules.mixedCase || (/\p{Lu}/u.test(value) && /\p{Ll}/u.test(value)),
			numbers: !rules.numbers || /\p{N}/u.test(value),
			symbols: !rules.symbols || /[^\p{L}\p{N}\s]/u.test(value),
			/* Only the server knows, so this line never ticks here. */
			uncompromised: false,
		};

		var total = 0;
		var satisfied = 0;
		each(root, '[data-requirement]', function (line) {
			var key = line.getAttribute('data-requirement');
			var ok = met[key] === true;
			total += 1;
			if (ok) satisfied += 1;
			line.setAttribute('data-met', ok ? 'true' : 'false');
			var done = line.querySelector('[data-part="done"]');
			if (done) done.hidden = !ok;
		});

		var fill = root.querySelector('[data-part="fill"]');
		if (fill) {
			fill.style.width = total === 0 ? '0%' : Math.round((satisfied / total) * 100) + '%';
		}
		var meter = root.querySelector('[data-part="meter"]');
		if (meter) {
			meter.setAttribute('aria-valuenow', String(satisfied));
			meter.setAttribute('aria-valuemax', String(total));
		}
		/* An empty box has nothing to say about itself, so the panel that
		 * explains the policy stays open and the strength summary stays quiet. */
		var strength = root.querySelector('[data-part="strength"]');
		if (strength) strength.setAttribute('data-met', total > 0 && satisfied === total ? 'true' : 'false');
	}

	/* ---- copy ---------------------------------------------------------------
	 *
	 * Puts a value on the clipboard and says so.
	 *
	 * The confirmation is written into a live region inside the button rather
	 * than swapped for its label, because the label is what the control is
	 * called and a control that renames itself after being pressed is a
	 * control a screen reader can no longer find by name. The region says
	 * "Copied" and empties again; the button keeps its name throughout.
	 *
	 * The writing itself is the asynchronous clipboard, and it is allowed only
	 * from a gesture and only on a secure origin. Both are true here -- this
	 * runs inside a click -- but a page served over plain HTTP has no
	 * clipboard at all, so the failure is answered rather than assumed away:
	 * the region says the copy did not happen, which is the truth and is more
	 * use than a tick that lies.
	 */
	arandu.ui.define('copy', {
		mounted: function (ctx) {
			var button = ctx.element;
			ctx.copy = function () {
				var props = ctx.props || {};
				var text = props.value;
				if (!text && props.source) {
					var from = document.getElementById(props.source);
					if (from) text = from.value !== undefined ? from.value : from.textContent;
				}
				if (!text) return;

				var say = function (message) {
					var region = button.querySelector('[data-part="feedback"]');
					if (!region) return;
					region.textContent = message;
					/* Emptied again so the next copy is a change, and a change
					 * is the only thing a live region announces. */
					window.setTimeout(function () { region.textContent = ''; }, 2000);
				};

				if (!navigator.clipboard || !navigator.clipboard.writeText) {
					say('Copying is not available here');
					return;
				}
				navigator.clipboard.writeText(text).then(
					function () { say(props.copied || 'Copied'); },
					function () { say('Copy failed'); }
				);
			};
			button.addEventListener('click', ctx.copy);
		},
		destroyed: function (ctx) {
			if (ctx.copy) ctx.element.removeEventListener('click', ctx.copy);
		},
	});

	/* ---- relative-time -------------------------------------------------------
	 *
	 * Rewrites a timestamp in the reader's own locale and time zone.
	 *
	 * The server cannot know either: no request carries them, and a server
	 * that guesses from the address is wrong for everyone travelling. So the
	 * server writes the instant in the datetime attribute, where it is exact,
	 * and a sentence in the element, where it is readable -- and this replaces
	 * the sentence once, with the browser's own formatter.
	 *
	 * Everything here degrades to what the server wrote: an unparseable
	 * instant, a browser without Intl, a script that never ran. The element is
	 * never emptied and never left saying "Invalid Date".
	 */
	arandu.ui.define('relative-time', {
		mounted: function (ctx) {
			var element = ctx.element;
			var when = new Date(element.getAttribute('datetime'));
			if (isNaN(when.getTime())) return;

			var style = (ctx.props || {}).style;
			var text = localised(when, style);
			if (text) element.textContent = text;
		},
	});

	/* localised writes one instant the way the reader's locale writes it, and
	 * returns nothing when it cannot -- which leaves the server's sentence. */
	function localised(when, style) {
		try {
			if (style === 'relative') {
				if (typeof Intl === 'undefined' || !Intl.RelativeTimeFormat) return '';
				return elapsed(when);
			}
			if (typeof Intl === 'undefined' || !Intl.DateTimeFormat) return '';
			if (style === 'time') return when.toLocaleTimeString();
			if (style === 'datetime') return when.toLocaleString();
			if (style === 'date') return when.toLocaleDateString();
			return '';
		} catch (error) {
			return '';
		}
	}

	/* elapsed says how long ago, in the largest unit that still has a whole
	 * number in it: "3 hours ago" and not "180 minutes ago".
	 *
	 * The table is walked from the largest down, so the first unit whose span
	 * fits is the one used. Months are 30 days and years are 365, which is
	 * wrong by up to a day and a half and is invisible at the scale where
	 * those units are chosen -- "2 months ago" does not become wrong because
	 * February is short.
	 */
	function elapsed(when) {
		var seconds = Math.round((when.getTime() - Date.now()) / 1000);
		var units = [
			['year', 31536000],
			['month', 2592000],
			['week', 604800],
			['day', 86400],
			['hour', 3600],
			['minute', 60],
		];
		var format = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
		for (var at = 0; at < units.length; at++) {
			var span = units[at][1];
			if (Math.abs(seconds) >= span) {
				return format.format(Math.round(seconds / span), units[at][0]);
			}
		}
		return format.format(seconds, 'second');
	}

	/* By window load every deferred script has run, so a behaviour still
	 * unmounted is one nobody registered -- which is worth one line each, and is
	 * the only moment this file can tell that apart from a script that has not
	 * run yet. From here on a miss is reported where it is found. */
	window.addEventListener('load', function () {
		loaded = true;
		mount(document);
		each(document, '[data-kyse-behavior]', function (element) {
			if (element.getAttribute('data-kyse-mounted') !== 'true') {
				miss('behaviour', element.getAttribute('data-kyse-behavior'));
			}
		});
	});
})();
