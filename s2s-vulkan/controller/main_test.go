package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	goruntime "runtime"
	"strings"
	"testing"
	"time"
)

func testController(t *testing.T, docker http.Handler) *controller {
	t.Helper()
	server := httptest.NewServer(docker)
	t.Cleanup(server.Close)
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "tcp", strings.TrimPrefix(server.URL, "http://"))
	}}
	runtime := runtimePolicy{
		BackendID: "parakeet-tdt-0.6b-v3", VariantID: "parakeet-cpu", Stage: "asr",
		Container: "s2s-parakeet-cpu", Image: "ghcr.io/antibyte/s2s-asr-parakeet@sha256:" + strings.Repeat("a", 64),
		Architectures: []string{goruntime.GOARCH}, Healthcheck: healthcheckPolicy{Path: "/health", Port: 8082},
	}
	return &controller{
		docker: &http.Client{Transport: transport}, policy: bundlePolicy{BundleVersion: "test", Runtimes: []runtimePolicy{runtime}},
		byVariant: map[string]runtimePolicy{runtime.VariantID: runtime}, byContainer: map[string][]runtimePolicy{runtime.Container: {runtime}},
		token: "test-token-that-is-long-enough-1234", network: "speech-lab", modelsVolume: "models", dataVolume: "data", bundleDigest: "digest",
	}
}

func TestDockerLayerProgressDeduplicatesEvents(t *testing.T) {
	layers := make(map[string]dockerLayerProgress)
	steps := []struct {
		id, status           string
		current, total, want int64
	}{
		{"layer-a", "Downloading", 40, 100, 40},
		{"layer-b", "Downloading", 20, 50, 60},
		{"layer-a", "Downloading", 30, 100, 60},
		{"layer-a", "Download complete", 0, 0, 120},
		{"layer-b", "Downloading", 50, 50, 150},
	}
	for _, step := range steps {
		if got := updateDockerLayerProgress(layers, step.id, step.status, step.current, step.total); got != step.want {
			t.Fatalf("layer progress = %d, want %d", got, step.want)
		}
	}
}

func TestModuleStatusReportsImagePullBytes(t *testing.T) {
	c := testController(t, http.NotFoundHandler())
	runtime := c.byVariant["parakeet-cpu"]
	runtime.ImageDownloadSizeBytes = 100
	c.byVariant[runtime.VariantID] = runtime
	c.imagePulls = map[string]int64{runtime.Image: 42}
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodGet, server.URL+"/s2s/modules/parakeet-cpu", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var status struct {
		ImageDownloadedBytes int64 `json:"image_downloaded_bytes"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if status.ImageDownloadedBytes != 42 {
		t.Fatalf("image bytes = %d, want 42", status.ImageDownloadedBytes)
	}
}

func TestRuntimeImagePullPublishesLiveBytes(t *testing.T) {
	firstEvent := make(chan struct{})
	release := make(chan struct{})
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/images/create") {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, "{\"id\":\"layer-a\",\"status\":\"Downloading\",\"progressDetail\":{\"current\":42,\"total\":100}}\n")
		w.(http.Flusher).Flush()
		close(firstEvent)
		<-release
		_, _ = io.WriteString(w, "{\"id\":\"layer-a\",\"status\":\"Download complete\"}\n")
	})
	c := testController(t, docker)
	image := c.byVariant["parakeet-cpu"].Image
	done := make(chan error, 1)
	go func() { done <- c.pullImage(context.Background(), image) }()
	select {
	case <-firstEvent:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("Docker pull did not stream its first event")
	}
	deadline := time.Now().Add(time.Second)
	for {
		c.pullMu.RLock()
		loaded := c.imagePulls[image]
		c.pullMu.RUnlock()
		if loaded == 42 {
			break
		}
		if time.Now().After(deadline) {
			close(release)
			t.Fatalf("live image bytes = %d, want 42", loaded)
		}
		time.Sleep(time.Millisecond)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestWaitForExistingTokenHandlesInitRace(t *testing.T) {
	path := t.TempDir() + "/token"
	go func() {
		time.Sleep(25 * time.Millisecond)
		_ = os.WriteFile(path, []byte(strings.Repeat("a", 64)), 0o600)
	}()
	token, err := waitForExistingToken(path, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if len(token) != 64 {
		t.Fatalf("token length = %d", len(token))
	}
}

func TestLoadOrCreateTokenPublishesReadOnlySharedFile(t *testing.T) {
	path := t.TempDir() + "/token"
	if _, err := loadOrCreateToken(path); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o444 {
		t.Fatalf("token mode = %o, want 444", got)
	}
}

func TestControllerRejectsUnauthenticatedAndUnknownTargets(t *testing.T) {
	c := testController(t, http.NotFoundHandler())
	server := httptest.NewServer(c.routes())
	defer server.Close()
	resp, err := http.Get(server.URL + "/containers/s2s-parakeet-cpu/json")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status = %d", resp.StatusCode)
	}
	req, _ := http.NewRequest(http.MethodGet, server.URL+"/containers/unrelated/json", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("unknown target status = %d", resp.StatusCode)
	}
	req, _ = http.NewRequest(http.MethodPost, server.URL+"/containers/s2s-parakeet-cpu/exec", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("unsupported action status = %d", resp.StatusCode)
	}
}

func TestStartRejectsOwnedNameWithImageDrift(t *testing.T) {
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/json") {
			_, _ = w.Write([]byte(`{"Id":"target","Name":"/s2s-parakeet-cpu","Config":{"Image":"ghcr.io/antibyte/foreign@sha256:` + strings.Repeat("b", 64) + `","Labels":{"aurago.managed":"speech-lab","aurago.role":"module","s2s.lab.managed":"true","backend-id":"parakeet-tdt-0.6b-v3","variant-id":"parakeet-cpu","stage":"asr","s2s.image":"ghcr.io/antibyte/foreign@sha256:` + strings.Repeat("b", 64) + `"}}}`))
			return
		}
		t.Fatalf("unexpected Docker call: %s %s", r.Method, r.URL.RequestURI())
	})
	c := testController(t, docker)
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/containers/s2s-parakeet-cpu/start", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("image drift status = %d", resp.StatusCode)
	}
}

func TestInstallCreatesOnlyPolicyRuntime(t *testing.T) {
	var created map[string]any
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/json"):
			http.NotFound(w, r)
		case strings.HasSuffix(r.URL.Path, "/images/create"):
			w.WriteHeader(http.StatusOK)
		case strings.HasSuffix(r.URL.Path, "/containers/create"):
			if err := json.NewDecoder(r.Body).Decode(&created); err != nil {
				t.Fatal(err)
			}
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{"Id":"created"}`))
		default:
			http.NotFound(w, r)
		}
	})
	c := testController(t, docker)
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/s2s/modules/parakeet-cpu/install", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("install status = %d", resp.StatusCode)
	}
	labels := created["Labels"].(map[string]any)
	if labels["backend-id"] != "parakeet-tdt-0.6b-v3" || labels["aurago.managed"] != ownerLabel {
		t.Fatalf("created labels = %#v", labels)
	}
	if _, ok := created["Entrypoint"]; ok {
		t.Fatal("caller-controlled entrypoint reached Docker create")
	}
	hostConfig := created["HostConfig"].(map[string]any)
	tmpfs := hostConfig["Tmpfs"].(map[string]any)
	if tmpfs["/tmp"] != "rw,nosuid,nodev,exec,size=268435456" {
		t.Fatalf("module tmpfs = %#v", tmpfs)
	}
	capDrop := hostConfig["CapDrop"].([]any)
	if len(capDrop) != 1 || capDrop[0] != "ALL" {
		t.Fatalf("module cap drop = %#v", capDrop)
	}
	capAdd := hostConfig["CapAdd"].([]any)
	if len(capAdd) != 2 || capAdd[0] != "SETUID" || capAdd[1] != "SETGID" {
		t.Fatalf("module cap add = %#v", capAdd)
	}
}

func TestInstallReusesExactImmutableImage(t *testing.T) {
	var pulled bool
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/json"):
			http.NotFound(w, r)
		case strings.Contains(r.URL.Path, "/images/") && strings.HasSuffix(r.URL.Path, "/json"):
			_, _ = w.Write([]byte(`{"Id":"sha256:present"}`))
		case strings.HasSuffix(r.URL.Path, "/images/create"):
			pulled = true
			w.WriteHeader(http.StatusOK)
		case strings.HasSuffix(r.URL.Path, "/containers/create"):
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{"Id":"created"}`))
		default:
			http.NotFound(w, r)
		}
	})
	c := testController(t, docker)
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/s2s/modules/parakeet-cpu/install", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("install status = %d", resp.StatusCode)
	}
	if pulled {
		t.Fatal("already-present immutable image was pulled again")
	}
}

func TestInstallReportsDockerPullErrorAfterLargeProgressStream(t *testing.T) {
	created := false
	progress := strings.Repeat("{\"status\":\"Downloading\"}\n", 1000)
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/json"):
			http.NotFound(w, r)
		case strings.Contains(r.URL.Path, "/images/") && strings.HasSuffix(r.URL.Path, "/json"):
			http.NotFound(w, r)
		case strings.HasSuffix(r.URL.Path, "/images/create"):
			for i := 0; i < 700; i++ {
				_, _ = io.WriteString(w, progress)
			}
			_, _ = io.WriteString(w, "{\"error\":\"registry refused pull\"}\n")
		case strings.HasSuffix(r.URL.Path, "/containers/create"):
			created = true
			w.WriteHeader(http.StatusCreated)
		default:
			http.NotFound(w, r)
		}
	})
	c := testController(t, docker)
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/s2s/modules/parakeet-cpu/install", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("install status = %d, want conflict", resp.StatusCode)
	}
	var result map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result["error"], "registry refused pull") {
		t.Fatalf("late pull error not reported: %q", result["error"])
	}
	if created {
		t.Fatal("runtime container created after Docker pull error")
	}
}

func TestRuntimePolicyRequiresDigestAndKnownStage(t *testing.T) {
	base := runtimePolicy{BackendID: "b", VariantID: "v", Stage: "asr", Container: "c", Image: "repo@sha256:" + strings.Repeat("a", 64), Architectures: []string{goruntime.GOARCH}, Healthcheck: healthcheckPolicy{Path: "/health", Port: 8082}}
	if err := validateRuntimePolicy(base); err != nil {
		t.Fatal(err)
	}
	bad := base
	bad.Image = "repo:latest"
	if err := validateRuntimePolicy(bad); err == nil {
		t.Fatal("mutable image accepted")
	}
	bad = base
	bad.Stage = "worker"
	if err := validateRuntimePolicy(bad); err == nil {
		t.Fatal("unknown stage accepted")
	}
}

func TestStartStopsRunningSiblingInSameStage(t *testing.T) {
	var calls []string
	docker := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.RequestURI())
		switch {
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/json"):
			_, _ = w.Write([]byte(`{"Id":"target","Name":"/s2s-parakeet-cpu","Config":{"Image":"ghcr.io/antibyte/s2s-asr-parakeet@sha256:` + strings.Repeat("a", 64) + `","Labels":{"aurago.managed":"speech-lab","aurago.role":"module","aurago.bundle":"test","aurago.fingerprint":"digest","s2s.lab.managed":"true","backend-id":"parakeet-tdt-0.6b-v3","variant-id":"parakeet-cpu","stage":"asr","s2s.image":"ghcr.io/antibyte/s2s-asr-parakeet@sha256:` + strings.Repeat("a", 64) + `"}}}`))
		case strings.HasSuffix(r.URL.Path, "/containers/json"):
			_, _ = w.Write([]byte(`[{"Id":"sibling","Names":["/s2s-whisper-cpu"],"Image":"ghcr.io/antibyte/s2s-asr-parakeet@sha256:` + strings.Repeat("a", 64) + `","Labels":{"aurago.managed":"speech-lab","aurago.role":"module","aurago.bundle":"test","aurago.fingerprint":"digest","s2s.lab.managed":"true","backend-id":"parakeet-tdt-0.6b-v3","variant-id":"whisper-cpu","stage":"asr","s2s.image":"ghcr.io/antibyte/s2s-asr-parakeet@sha256:` + strings.Repeat("a", 64) + `"}}]`))
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-whisper-cpu/stop"):
			w.WriteHeader(http.StatusNoContent)
		case strings.HasSuffix(r.URL.Path, "/containers/s2s-parakeet-cpu/start"):
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	})
	c := testController(t, docker)
	sibling := c.policy.Runtimes[0]
	sibling.VariantID = "whisper-cpu"
	sibling.Container = "s2s-whisper-cpu"
	c.policy.Runtimes = append(c.policy.Runtimes, sibling)
	c.byVariant[sibling.VariantID] = sibling
	c.byContainer[sibling.Container] = []runtimePolicy{sibling}
	server := httptest.NewServer(c.routes())
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/containers/s2s-parakeet-cpu/start", nil)
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("start status = %d, calls = %#v", resp.StatusCode, calls)
	}
	wantStop := "POST /containers/s2s-whisper-cpu/stop?t=10"
	wantStart := "POST /containers/s2s-parakeet-cpu/start"
	if strings.Join(calls, "\n") != strings.Join([]string{
		"GET /v1.44/containers/s2s-parakeet-cpu/json",
		"GET /v1.44/containers/json?all=0",
		"POST /v1.44" + strings.TrimPrefix(wantStop, "POST "),
		"POST /v1.44" + strings.TrimPrefix(wantStart, "POST "),
	}, "\n") {
		t.Fatalf("Docker calls = %#v", calls)
	}
}
