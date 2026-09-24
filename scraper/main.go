package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

type record struct {
	URL         string `json:"url"`
	SourceURL   string `json:"source_url,omitempty"`
	Title       string `json:"title"`
	Text        string `json:"text"`
	FetchedAt   string `json:"fetched_at"`
	ContentHash string `json:"content_hash"`
}

type page struct {
	url   string
	depth int
}

var (
	tagRE    = regexp.MustCompile(`(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</[a-z]+>`)
	allTagRE = regexp.MustCompile(`<[^>]+>`)
	linkRE   = regexp.MustCompile(`(?i)href\s*=\s*["']([^"']+)["']`)
	titleRE  = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	spaceRE  = regexp.MustCompile(`\s+`)
	startRE  = regexp.MustCompile(`(?i)\*\*\*\s*START OF[^*\r\n]*PROJECT GUTENBERG[^*\r\n]*\*\*\*`)
	endRE    = regexp.MustCompile(`(?i)\*\*\*\s*END OF[^*\r\n]*PROJECT GUTENBERG[^*\r\n]*\*\*\*`)
)

func main() {
	seeds := flag.String("seeds", "", "comma-separated HTTP(S) seed URLs (required)")
	allow := flag.String("allow", "", "comma-separated allowed hostnames (required)")
	out := flag.String("out", "data/raw.jsonl", "JSONL output path")
	maxPages := flag.Int("max-pages", 200, "maximum pages to fetch")
	maxDepth := flag.Int("max-depth", 2, "maximum link depth")
	delay := flag.Duration("delay", time.Second, "minimum delay between requests to a host")
	flag.Parse()
	if *seeds == "" || *allow == "" {
		log.Fatal("-seeds and -allow are required; do not run an unbounded crawl")
	}

	allowed := map[string]bool{}
	for _, host := range strings.Split(*allow, ",") {
		allowed[strings.ToLower(strings.TrimSpace(host))] = true
	}
	queue := []page{}
	for _, raw := range strings.Split(*seeds, ",") {
		u, err := url.Parse(strings.TrimSpace(raw))
		if err != nil || u.Scheme != "http" && u.Scheme != "https" {
			log.Fatalf("invalid seed %q", raw)
		}
		queue = append(queue, page{url: u.String()})
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	writer := json.NewEncoder(f)
	client := &http.Client{Timeout: 20 * time.Second, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 3 {
			return fmt.Errorf("too many redirects")
		}
		return nil
	}}
	seen, contentSeen := map[string]bool{}, map[string]bool{}
	robots := map[string][]string{}
	lastRequest := map[string]time.Time{}

	for len(queue) > 0 && len(seen) < *maxPages {
		current := queue[0]
		queue = queue[1:]
		u, err := url.Parse(current.url)
		if err != nil {
			continue
		}
		u.Fragment = ""
		current.url = u.String()
		if seen[current.url] || !allowed[strings.ToLower(u.Hostname())] {
			continue
		}
		seen[current.url] = true
		host := strings.ToLower(u.Host)
		if _, ok := robots[host]; !ok {
			robots[host] = fetchRobots(client, u)
		}
		if !permitted(u.Path, robots[host]) {
			log.Printf("robots denied %s", current.url)
			continue
		}
		waitHost(host, *delay, lastRequest)
		resp, err := getWithRetry(client, current.url)
		if err != nil {
			log.Printf("fetch %s: %v", current.url, err)
			continue
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if readErr != nil || resp.StatusCode < 200 || resp.StatusCode >= 300 {
			log.Printf("skip %s: bad response", current.url)
			continue
		}
		contentType := strings.ToLower(resp.Header.Get("Content-Type"))
		var text, title string
		sourceURL := ""
		if isPlainTextURL(u.String()) || isPlainTextContentType(contentType) {
			text = cleanText(string(body))
		} else if strings.Contains(contentType, "text/html") {
			text, title = extract(string(body))
			if isLandingJunk(text) {
				log.Printf("skip %s: non-prose boilerplate", current.url)
				continue
			}
			textURL := preferredTextLink(u, string(body), allowed)
			if textURL != "" {
				waitHost(strings.ToLower(mustURLHost(textURL)), *delay, lastRequest)
				textResp, textErr := getWithRetry(client, textURL)
				if textErr == nil {
					textBody, textReadErr := io.ReadAll(io.LimitReader(textResp.Body, 4<<20))
					textResp.Body.Close()
					if textReadErr == nil && textResp.StatusCode >= 200 && textResp.StatusCode < 300 && !strings.Contains(strings.ToLower(textResp.Header.Get("Content-Type")), "text/html") {
						text = cleanText(string(textBody))
						sourceURL = textURL
					}
				}
			}
		} else {
			continue
		}
		if len(text) < 80 || isSignInJunk(text) {
			continue
		}
		h := sha256.Sum256([]byte(text))
		hash := hex.EncodeToString(h[:])
		if contentSeen[hash] {
			continue
		}
		contentSeen[hash] = true
		if err := writer.Encode(record{URL: current.url, SourceURL: sourceURL, Title: title, Text: text, FetchedAt: time.Now().UTC().Format(time.RFC3339), ContentHash: hash}); err != nil {
			log.Fatal(err)
		}
		if current.depth >= *maxDepth {
			continue
		}
		for _, href := range linkRE.FindAllStringSubmatch(string(body), -1) {
			if next, ok := sameDomain(u, href[1], allowed); ok && !seen[next] {
				queue = append(queue, page{url: next, depth: current.depth + 1})
			}
		}
	}
	log.Printf("wrote %d pages to %s", len(contentSeen), *out)
}

func extract(html string) (string, string) {
	title := ""
	if match := titleRE.FindStringSubmatch(html); len(match) > 1 {
		title = clean(match[1])
	}
	text := cleanText(allTagRE.ReplaceAllString(tagRE.ReplaceAllString(html, " "), " "))
	return text, title
}
func clean(value string) string {
	value = html.UnescapeString(value)
	value = strings.ReplaceAll(value, "\u00a0", " ")
	return strings.TrimSpace(spaceRE.ReplaceAllString(value, " "))
}
func cleanText(value string) string {
	return clean(stripGutenbergBoilerplate(value))
}

func stripGutenbergBoilerplate(value string) string {
	start := startRE.FindStringIndex(value)
	if start == nil {
		if end := endRE.FindStringIndex(value); end != nil {
			return value[:end[0]]
		}
		return value
	}
	value = value[start[1]:]
	if end := endRE.FindStringIndex(value); end != nil {
		return value[:end[0]]
	}
	return value
}

func preferredTextLink(base *url.URL, body string, allowed map[string]bool) string {
	best, bestScore := "", -1
	for _, match := range linkRE.FindAllStringSubmatch(body, -1) {
		next, ok := sameDomain(base, match[1], allowed)
		if !ok || !isPlainTextURL(next) || strings.Contains(strings.ToLower(next), "/ebooks/send/") {
			continue
		}
		score := 1
		lower := strings.ToLower(next)
		if strings.Contains(lower, "-0.txt") {
			score++
		}
		if strings.Contains(lower, "utf-8") || strings.Contains(lower, "utf8") {
			score++
		}
		if score > bestScore {
			best, bestScore = next, score
		}
	}
	return best
}

func isPlainTextURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	path := strings.ToLower(u.Path)
	return strings.HasSuffix(path, ".txt") || strings.HasSuffix(path, ".txt.utf-8") || strings.HasSuffix(path, ".txt.utf8")
}

func isPlainTextContentType(contentType string) bool {
	return strings.Contains(contentType, "text/plain") || strings.Contains(contentType, "application/octet-stream")
}

var landingJunkRE = regexp.MustCompile(`(?i)(Project Gutenberg 79,270 free eBooks|Reading Options &amp;? Kindle|Frequently Downloaded|Readers also downloaded|Displaying results \d+[-–]\d+|Sign in with Google|downloads?\s*\d+|Of the Project Gutenberg|About Project Gutenberg|Main Categories|Reading Lists|Search Options|Index of /files)`)

func isLandingJunk(text string) bool {
	return landingJunkRE.MatchString(text) && len(text) < 40000
}

func isSignInJunk(text string) bool {
	return strings.Contains(text, "Sign in with Google") && strings.Contains(text, "Forgot email")
}

func waitHost(host string, delay time.Duration, lastRequest map[string]time.Time) {
	if wait := delay - time.Since(lastRequest[host]); wait > 0 {
		time.Sleep(wait)
	}
	lastRequest[host] = time.Now()
}

func mustURLHost(raw string) string {
	u, _ := url.Parse(raw)
	return u.Host
}

func getWithRetry(client *http.Client, rawURL string) (*http.Response, error) {
	const attempts = 3
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		req, err := http.NewRequest(http.MethodGet, rawURL, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", "skynet-research-crawler/0.1 (+contact required)")
		resp, err := client.Do(req)
		if err == nil && !transientStatus(resp.StatusCode) {
			return resp, nil
		}
		if resp != nil {
			resp.Body.Close()
			lastErr = fmt.Errorf("HTTP status %d", resp.StatusCode)
		} else {
			lastErr = err
		}
		if attempt+1 < attempts {
			time.Sleep(time.Duration(100*(1<<attempt)) * time.Millisecond)
		}
	}
	return nil, lastErr
}

func transientStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooEarly || status == http.StatusTooManyRequests || status >= 500
}
func sameDomain(base *url.URL, href string, allowed map[string]bool) (string, bool) {
	next, err := base.Parse(href)
	if err != nil || next.Scheme != "http" && next.Scheme != "https" || !allowed[strings.ToLower(next.Hostname())] {
		return "", false
	}
	next.Fragment = ""
	next.RawQuery = ""
	return next.String(), true
}
func fetchRobots(client *http.Client, base *url.URL) []string {
	robotsURL := *base
	robotsURL.Path = "/robots.txt"
	resp, err := getWithRetry(client, robotsURL.String())
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil
	}
	scanner := bufio.NewScanner(io.LimitReader(resp.Body, 256<<10))
	var rules []string
	active := false
	for scanner.Scan() {
		line := strings.TrimSpace(strings.SplitN(scanner.Text(), "#", 2)[0])
		lower := strings.ToLower(line)
		if strings.HasPrefix(lower, "user-agent:") {
			active = strings.TrimSpace(strings.TrimPrefix(lower, "user-agent:")) == "*"
		}
		if active && strings.HasPrefix(lower, "disallow:") {
			path := strings.TrimSpace(line[len("disallow:"):])
			if path != "" {
				rules = append(rules, path)
			}
		}
	}
	sort.Strings(rules)
	return rules
}
func permitted(path string, rules []string) bool {
	for _, rule := range rules {
		if strings.HasPrefix(path, rule) {
			return false
		}
	}
	return true
}
