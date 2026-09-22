package middleware

import (
	"net/http"
	"strconv"
)

// ValidatePostSize checks the Content-Length header against maxSize and
// returns 413 Request Entity Too Large if the body exceeds the limit.
func ValidatePostSize(maxSize int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.ContentLength > maxSize {
				http.Error(w, "Request Entity Too Large", http.StatusRequestEntityTooLarge)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// ValidatePathEncoding validates that the request path is valid UTF-8. An
// invalid path returns a 400 Bad Request.
func ValidatePathEncoding() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Go's net/http already validates UTF-8 in URLs; this is
			// a defense-in-depth check.
			path := r.URL.EscapedPath()
			for i := 0; i < len(path); i++ {
				if path[i] >= 0x80 {
					http.Error(w, "Malformed URL", http.StatusBadRequest)
					return
				}
			}
			next.ServeHTTP(w, r)
		})
	}
}

// SetCacheHeaders sets cache headers (ETag, Last-Modified, Cache-Control) on
// the response. The options map can contain "etag", "last_modified",
// "max_age", "s_maxage", "private", and "public" keys.
func SetCacheHeaders(options map[string]string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			h := w.Header()

			if etag, ok := options["etag"]; ok {
				h.Set("ETag", etag)
				if match := r.Header.Get("If-None-Match"); match == etag {
					w.WriteHeader(http.StatusNotModified)
					return
				}
			}

			if lastMod, ok := options["last_modified"]; ok {
				h.Set("Last-Modified", lastMod)
			}

			cacheControl := ""
			if maxAge, ok := options["max_age"]; ok {
				cacheControl += "max-age=" + maxAge
			}
			if sMaxAge, ok := options["s_maxage"]; ok {
				if cacheControl != "" {
					cacheControl += ", "
				}
				cacheControl += "s-maxage=" + sMaxAge
			}
			if _, ok := options["public"]; ok {
				if cacheControl != "" {
					cacheControl += ", "
				}
				cacheControl += "public"
			}
			if _, ok := options["private"]; ok {
				if cacheControl != "" {
					cacheControl += ", "
				}
				cacheControl += "private"
			}
			if cacheControl != "" {
				h.Set("Cache-Control", cacheControl)
			}

			next.ServeHTTP(w, r)
		})
	}
}

// CheckResponseForModifications checks If-None-Match and If-Modified-Since
// headers and returns 304 Not Modified if the response has not changed.
func CheckResponseForModifications(etag string, lastModified string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if etag != "" {
				if match := r.Header.Get("If-None-Match"); match == etag {
					w.WriteHeader(http.StatusNotModified)
					return
				}
			}
			if lastModified != "" {
				if modSince := r.Header.Get("If-Modified-Since"); modSince == lastModified {
					w.WriteHeader(http.StatusNotModified)
					return
				}
			}
			next.ServeHTTP(w, r)
		})
	}
}

// TrustHosts validates that the request Host header matches the list of
// trusted host patterns. A request with a non-matching host is rejected
// with a 400 Bad Request.
//
// Hosts named through [At] are trusted as well, and [At] can turn subdomain
// matching on for them.
func TrustHosts(hosts []string, subdomains bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			trusted := hosts
			matchSubdomains := subdomains
			if always := Hosts(); len(always) > 0 {
				trusted = append(append([]string{}, hosts...), always...)
				matchSubdomains = matchSubdomains || TrustsSubdomains()
			}
			hostSet := make(map[string]bool, len(trusted))
			for _, h := range trusted {
				hostSet[h] = true
			}

			host := r.Host
			if hostSet[host] {
				next.ServeHTTP(w, r)
				return
			}
			if matchSubdomains {
				for trusted := range hostSet {
					if len(host) > len(trusted) && host[len(host)-len(trusted)-1] == '.' && host[len(host)-len(trusted):] == trusted {
						next.ServeHTTP(w, r)
						return
					}
				}
			}
			http.Error(w, "Bad Request: Invalid Host", http.StatusBadRequest)
		})
	}
}

// HandleCors handles CORS preflight requests and adds CORS headers to
// responses. The allowedOrigins, allowedMethods, and allowedHeaders
// parameters configure which origins, methods, and headers are permitted.
func HandleCors(allowedOrigins []string, allowedMethods []string, allowedHeaders []string, maxAge int, allowCredentials bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if shouldSkipCors(r) {
				next.ServeHTTP(w, r)
				return
			}

			origin := r.Header.Get("Origin")
			if origin == "" {
				next.ServeHTTP(w, r)
				return
			}

			// The wildcard answers "*", and a named origin answers itself.
			//
			// Echoing the caller's own origin for a wildcard is what makes a
			// permissive list readable by everybody WITH credentials, because it
			// slips past the browser check that "*" plus credentials fails. The
			// two are kept apart here, and CorsConfig.Valid refuses the
			// combination at boot so this branch is never reached with both.
			wildcard := false
			allowed := false
			for _, o := range allowedOrigins {
				if o == "*" {
					wildcard, allowed = true, true
					break
				}
				if o == origin {
					allowed = true
					break
				}
			}
			if !allowed {
				next.ServeHTTP(w, r)
				return
			}

			h := w.Header()
			if wildcard {
				h.Set("Access-Control-Allow-Origin", "*")
			} else {
				h.Set("Access-Control-Allow-Origin", origin)
			}
			// Never with the wildcard. A response carrying both is refused by
			// every browser, and the way people work around that is the leak
			// this middleware used to have.
			if allowCredentials && !wildcard {
				h.Set("Access-Control-Allow-Credentials", "true")
			}
			h.Set("Vary", "Origin")

			if r.Method == http.MethodOptions {
				// Preflight.
				if len(allowedMethods) > 0 {
					h.Set("Access-Control-Allow-Methods", joinHeaders(allowedMethods))
				}
				if len(allowedHeaders) > 0 {
					h.Set("Access-Control-Allow-Headers", joinHeaders(allowedHeaders))
				}
				if maxAge > 0 {
					h.Set("Access-Control-Max-Age", strconv.Itoa(maxAge))
				}
				w.WriteHeader(http.StatusNoContent)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

func joinHeaders(values []string) string {
	result := ""
	for i, v := range values {
		if i > 0 {
			result += ", "
		}
		result += v
	}
	return result
}
