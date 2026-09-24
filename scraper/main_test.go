package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
)

func TestExtractCleansEntitiesAndGutenbergMarkers(t *testing.T) {
	text, title := extract(`<title>Tom &amp; Jerry</title><p>*** START OF THE PROJECT GUTENBERG EBOOK ***</p><p>Tom&nbsp;&amp; Jerry &#8212; a story.</p><p>*** END OF THE PROJECT GUTENBERG EBOOK ***</p>`)
	if title != "Tom & Jerry" {
		t.Fatalf("title = %q", title)
	}
	if text != "Tom & Jerry \u2014 a story." {
		t.Fatalf("text = %q", text)
	}
}

func TestPreferredTextLink(t *testing.T) {
	base, _ := url.Parse("https://www.gutenberg.org/ebooks/1342")
	body := `<a href="/files/1342/1342.txt">plain text</a><a href="/files/1342/1342-0.txt">UTF-8</a>`
	got := preferredTextLink(base, body, map[string]bool{"www.gutenberg.org": true})
	want := "https://www.gutenberg.org/files/1342/1342-0.txt"
	if got != want {
		t.Fatalf("preferredTextLink = %q, want %q", got, want)
	}
}

func TestGetWithRetryRetriesTransientStatus(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, "ok")
	}))
	defer server.Close()

	resp, err := getWithRetry(server.Client(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || requests.Load() != 3 {
		t.Fatalf("status = %d, requests = %d", resp.StatusCode, requests.Load())
	}
}

func TestStripGutenbergBoilerplateWithoutEndMarker(t *testing.T) {
	got := cleanText(strings.Join([]string{
		"header",
		"*** START OF THE PROJECT GUTENBERG EBOOK ***",
		"body",
	}, "\n"))
	if got != "body" {
		t.Fatalf("got %q", got)
	}
}
