package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	goruntime "runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	ownerLabel   = "speech-lab"
	dockerAPIVer = "v1.44"
)

var safeID = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)
var immutableImage = regexp.MustCompile(`^[^\s@]+@sha256:[a-f0-9]{64}$`)

type bundlePolicy struct {
	BundleVersion string          `json:"bundle_version"`
	Runtimes      []runtimePolicy `json:"runtimes"`
}

type runtimePolicy struct {
	BackendID              string            `json:"backend_id"`
	VariantID              string            `json:"variant_id"`
	Stage                  string            `json:"stage"`
	Container              string            `json:"container"`
	Image                  string            `json:"image"`
	ImageDownloadSizeBytes int64             `json:"image_download_size_bytes,omitempty"`
	Command                []string          `json:"command,omitempty"`
	Environment            map[string]string `json:"environment,omitempty"`
	Aliases                []string          `json:"aliases,omitempty"`
	ModelMounts            []string          `json:"model_mounts,omitempty"`
	Accelerator            string            `json:"accelerator,omitempty"`
	ShmSizeBytes           int64             `json:"shm_size_bytes,omitempty"`
	Architectures          []string          `json:"architectures"`
	Experimental           bool              `json:"experimental,omitempty"`
	Healthcheck            healthcheckPolicy `json:"healthcheck"`
	Resources              resourcePolicy    `json:"resources"`
	License                string            `json:"license,omitempty"`
	AuthRequired           bool              `json:"auth_required,omitempty"`
}

type healthcheckPolicy struct {
	Path            string `json:"path"`
	Port            int    `json:"port"`
	TimeoutSeconds  int    `json:"timeout_seconds"`
	IntervalSeconds int    `json:"interval_seconds"`
}

type resourcePolicy struct {
	MemoryBytes int64   `json:"memory_bytes,omitempty"`
	NanoCPUs    int64   `json:"nano_cpus,omitempty"`
	ShmBytes    int64   `json:"shm_bytes,omitempty"`
	GPUCount    int64   `json:"gpu_count,omitempty"`
	VRAMGB      float64 `json:"vram_gb,omitempty"`
}

type controller struct {
	docker       *http.Client
	policy       bundlePolicy
	byVariant    map[string]runtimePolicy
	byContainer  map[string][]runtimePolicy
	token        string
	network      string
	modelsVolume string
	dataVolume   string
	bundleDigest string
	pullMu       sync.RWMutex
	imagePulls   map[string]int64
}

type dockerInspect struct {
	ID     string `json:"Id"`
	Name   string `json:"Name"`
	Config struct {
		Image  string            `json:"Image"`
		Labels map[string]string `json:"Labels"`
	} `json:"Config"`
	State struct {
		Running bool `json:"Running"`
	} `json:"State"`
}

func main() {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("S2S_CONTROLLER_INIT_TOKEN_ONLY")), "true") {
		if _, err := loadOrCreateToken(os.Getenv("S2S_CONTROLLER_TOKEN_FILE")); err != nil {
			log.Fatal(err)
		}
		return
	}
	c, err := newControllerFromEnv()
	if err != nil {
		log.Fatal(err)
	}
	addr := strings.TrimSpace(os.Getenv("S2S_CONTROLLER_ADDR"))
	if addr == "" {
		addr = "0.0.0.0:2375"
	}
	server := &http.Server{
		Addr:              addr,
		Handler:           c.routes(),
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       30 * time.Second,
	}
	log.Printf("Speech Lab controller listening on %s with %d runtime policies", addr, len(c.byVariant))
	log.Fatal(server.ListenAndServe())
}

func newControllerFromEnv() (*controller, error) {
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(os.Getenv("S2S_CONTROLLER_POLICY_B64")))
	if err != nil || len(raw) == 0 {
		return nil, fmt.Errorf("decode S2S_CONTROLLER_POLICY_B64: %w", err)
	}
	var policy bundlePolicy
	if err := json.Unmarshal(raw, &policy); err != nil {
		return nil, fmt.Errorf("parse controller policy: %w", err)
	}
	var token string
	if strings.EqualFold(strings.TrimSpace(os.Getenv("S2S_CONTROLLER_TOKEN_READ_ONLY")), "true") {
		token, err = waitForExistingToken(os.Getenv("S2S_CONTROLLER_TOKEN_FILE"), 15*time.Second)
	} else {
		token, err = loadOrCreateToken(os.Getenv("S2S_CONTROLLER_TOKEN_FILE"))
	}
	if err != nil {
		return nil, err
	}
	docker := &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{Timeout: 5 * time.Second}).DialContext(ctx, "unix", "/var/run/docker.sock")
		},
	}, Timeout: 0}
	c := &controller{
		docker:       docker,
		policy:       policy,
		byVariant:    map[string]runtimePolicy{},
		byContainer:  map[string][]runtimePolicy{},
		token:        token,
		network:      envOr("S2S_CONTROLLER_NETWORK", "aurago-speech-lab"),
		modelsVolume: envOr("S2S_CONTROLLER_MODELS_VOLUME", "aurago-speech-lab-models"),
		dataVolume:   envOr("S2S_CONTROLLER_DATA_VOLUME", "aurago-speech-lab-data"),
		bundleDigest: strings.TrimSpace(os.Getenv("S2S_CONTROLLER_BUNDLE_DIGEST")),
	}
	for _, runtime := range policy.Runtimes {
		if err := validateRuntimePolicy(runtime); err != nil {
			return nil, err
		}
		if _, exists := c.byVariant[runtime.VariantID]; exists {
			return nil, fmt.Errorf("duplicate runtime variant %q", runtime.VariantID)
		}
		c.byVariant[runtime.VariantID] = runtime
		c.byContainer[runtime.Container] = append(c.byContainer[runtime.Container], runtime)
	}
	if len(c.byVariant) == 0 {
		return nil, errors.New("controller policy has no runtimes")
	}
	return c, nil
}

func waitForExistingToken(path string, timeout time.Duration) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		path = "/control/token"
	}
	deadline := time.Now().Add(timeout)
	for {
		if data, err := os.ReadFile(path); err == nil {
			if token := strings.TrimSpace(string(data)); len(token) >= 32 {
				return token, nil
			}
		}
		if time.Now().After(deadline) {
			return "", fmt.Errorf("controller token file %q was not initialized", path)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func validateRuntimePolicy(runtime runtimePolicy) error {
	for field, value := range map[string]string{
		"backend_id": runtime.BackendID,
		"variant_id": runtime.VariantID,
		"container":  runtime.Container,
	} {
		if !safeID.MatchString(value) {
			return fmt.Errorf("runtime %s %q is invalid", field, value)
		}
	}
	if runtime.Stage != "asr" && runtime.Stage != "tts" && runtime.Stage != "llm" {
		return fmt.Errorf("runtime %q has invalid stage %q", runtime.VariantID, runtime.Stage)
	}
	if !immutableImage.MatchString(runtime.Image) {
		return fmt.Errorf("runtime %q image must use an immutable sha256 digest", runtime.VariantID)
	}
	if len(runtime.Architectures) == 0 {
		return fmt.Errorf("runtime %q has no published architectures", runtime.VariantID)
	}
	architectureOK := false
	for _, architecture := range runtime.Architectures {
		if architecture == goruntime.GOARCH {
			architectureOK = true
		}
	}
	if !architectureOK {
		return fmt.Errorf("runtime %q is not published for architecture %s", runtime.VariantID, goruntime.GOARCH)
	}
	if runtime.Healthcheck.Path == "" || !strings.HasPrefix(runtime.Healthcheck.Path, "/") || runtime.Healthcheck.Port < 1 || runtime.Healthcheck.Port > 65535 {
		return fmt.Errorf("runtime %q has an invalid healthcheck", runtime.VariantID)
	}
	for key := range runtime.Environment {
		if key == "" || strings.ContainsAny(key, "=\x00\r\n") {
			return fmt.Errorf("runtime %q has invalid environment key", runtime.VariantID)
		}
	}
	return nil
}

func loadOrCreateToken(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		path = "/control/token"
	}
	if data, err := os.ReadFile(path); err == nil {
		if token := strings.TrimSpace(string(data)); len(token) >= 32 {
			if err := os.Chmod(path, 0o444); err != nil {
				return "", fmt.Errorf("make controller token read-only: %w", err)
			}
			return token, nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", fmt.Errorf("create controller token directory: %w", err)
	}
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generate controller token: %w", err)
	}
	token := hex.EncodeToString(buf)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(token+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("write controller token: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return "", fmt.Errorf("publish controller token: %w", err)
	}
	if err := os.Chmod(path, 0o444); err != nil {
		return "", fmt.Errorf("make controller token read-only: %w", err)
	}
	return token, nil
}

func (c *controller) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "runtimes": len(c.byVariant)})
	})
	mux.HandleFunc("GET /containers/{container}/json", c.auth(c.inspectContainer))
	mux.HandleFunc("GET /containers/json", c.auth(c.listContainers))
	mux.HandleFunc("POST /containers/{container}/start", c.auth(c.startContainer))
	mux.HandleFunc("POST /containers/{container}/stop", c.auth(c.stopContainer))
	mux.HandleFunc("POST /s2s/modules/{variant}/install", c.auth(c.installModule))
	mux.HandleFunc("GET /s2s/modules/{variant}", c.auth(c.getModule))
	mux.HandleFunc("DELETE /s2s/modules/{variant}", c.auth(c.deleteModule))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		mux.ServeHTTP(w, r)
	})
}

func (c *controller) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
		if len(got) != len(c.token) || subtle.ConstantTimeCompare([]byte(got), []byte(c.token)) != 1 {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
		next(w, r)
	}
}

func (c *controller) inspectContainer(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("container")
	if _, ok := c.byContainer[name]; !ok {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "container is not allowlisted"})
		return
	}
	status, body, err := c.dockerRequest(r.Context(), http.MethodGet, "/containers/"+url.PathEscape(name)+"/json", nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if status == http.StatusNotFound {
		writeJSON(w, status, map[string]string{"error": "module container does not exist"})
		return
	}
	if status < 200 || status >= 300 {
		writeDockerError(w, status, body)
		return
	}
	var inspected dockerInspect
	if err := json.Unmarshal(body, &inspected); err != nil || !c.matchesAny(inspected) {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "container is not an owned Speech Lab module"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func (c *controller) listContainers(w http.ResponseWriter, r *http.Request) {
	status, body, err := c.dockerRequest(r.Context(), http.MethodGet, "/containers/json?all=1", nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if status < 200 || status >= 300 {
		writeDockerError(w, status, body)
		return
	}
	var rows []map[string]any
	if err := json.Unmarshal(body, &rows); err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "invalid Docker response"})
		return
	}
	filtered := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		labels, _ := row["Labels"].(map[string]any)
		if labelString(labels, "aurago.managed") != ownerLabel || labelString(labels, "s2s.lab.managed") != "true" {
			continue
		}
		names, _ := row["Names"].([]any)
		if len(names) == 0 {
			continue
		}
		name, _ := names[0].(string)
		name = strings.TrimPrefix(name, "/")
		image, _ := row["Image"].(string)
		if c.matchesListed(name, image, labels) {
			filtered = append(filtered, row)
		}
	}
	writeJSON(w, http.StatusOK, filtered)
}

func (c *controller) startContainer(w http.ResponseWriter, r *http.Request) {
	c.containerAction(w, r, "start")
}

func (c *controller) stopContainer(w http.ResponseWriter, r *http.Request) {
	timeout := 10
	if value := r.URL.Query().Get("t"); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil && parsed >= 1 && parsed <= 30 {
			timeout = parsed
		}
	}
	c.containerAction(w, r, "stop?t="+strconv.Itoa(timeout))
}

func (c *controller) containerAction(w http.ResponseWriter, r *http.Request, action string) {
	name := r.PathValue("container")
	inspected, found, err := c.inspectOwned(r.Context(), name)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if !found {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "module container does not exist"})
		return
	}
	if !c.matchesAny(inspected) {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "container does not match the signed runtime policy"})
		return
	}
	if action == "start" {
		if err := c.stopRunningStageSiblings(r.Context(), inspected); err != nil {
			writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
	}
	status, body, err := c.dockerRequest(r.Context(), http.MethodPost, "/containers/"+url.PathEscape(name)+"/"+action, nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if (status >= 200 && status < 300) || status == http.StatusNotModified {
		w.WriteHeader(status)
		return
	}
	writeDockerError(w, status, body)
}

func (c *controller) stopRunningStageSiblings(ctx context.Context, target dockerInspect) error {
	stage := target.Config.Labels["stage"]
	if stage == "" {
		return errors.New("target module has no stage label")
	}
	status, body, err := c.dockerRequest(ctx, http.MethodGet, "/containers/json?all=0", nil)
	if err != nil {
		return fmt.Errorf("list running stage modules: %w", err)
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("list running stage modules returned HTTP %d", status)
	}
	var rows []struct {
		ID     string            `json:"Id"`
		Names  []string          `json:"Names"`
		Image  string            `json:"Image"`
		Labels map[string]string `json:"Labels"`
	}
	if err := json.Unmarshal(body, &rows); err != nil {
		return fmt.Errorf("parse running stage modules: %w", err)
	}
	for _, row := range rows {
		if row.ID == target.ID || row.Labels["aurago.managed"] != ownerLabel ||
			row.Labels["s2s.lab.managed"] != "true" || row.Labels["stage"] != stage {
			continue
		}
		name := ""
		if len(row.Names) > 0 {
			name = strings.TrimPrefix(row.Names[0], "/")
		}
		labels := make(map[string]any, len(row.Labels))
		for key, value := range row.Labels {
			labels[key] = value
		}
		if !c.matchesListed(name, row.Image, labels) {
			return fmt.Errorf("running stage sibling %q does not match the signed runtime policy", name)
		}
		stopStatus, stopBody, stopErr := c.dockerRequest(ctx, http.MethodPost, "/containers/"+url.PathEscape(name)+"/stop?t=10", nil)
		if stopErr != nil {
			return fmt.Errorf("stop running stage sibling %q: %w", name, stopErr)
		}
		if (stopStatus < 200 || stopStatus >= 300) && stopStatus != http.StatusNotModified {
			return fmt.Errorf("stop running stage sibling %q: HTTP %d %s", name, stopStatus, strings.TrimSpace(string(stopBody)))
		}
	}
	return nil
}

func (c *controller) installModule(w http.ResponseWriter, r *http.Request) {
	runtime, ok := c.byVariant[r.PathValue("variant")]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "runtime variant is not allowlisted"})
		return
	}
	if err := c.ensureModule(r.Context(), runtime); err != nil {
		writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{
		"variant_id":                runtime.VariantID,
		"container":                 runtime.Container,
		"state":                     "ready",
		"image":                     runtime.Image,
		"image_download_size_bytes": runtime.ImageDownloadSizeBytes,
	})
}

func (c *controller) getModule(w http.ResponseWriter, r *http.Request) {
	runtime, ok := c.byVariant[r.PathValue("variant")]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "runtime variant is not allowlisted"})
		return
	}
	inspected, found, err := c.inspectOwned(r.Context(), runtime.Container)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	state := "missing"
	if found && c.matches(inspected, runtime) {
		state = "ready"
		if inspected.State.Running {
			state = "running"
		}
	}
	c.pullMu.RLock()
	downloaded := c.imagePulls[runtime.Image]
	c.pullMu.RUnlock()
	if state == "ready" || state == "running" {
		downloaded = runtime.ImageDownloadSizeBytes
	}
	if runtime.ImageDownloadSizeBytes > 0 && downloaded > runtime.ImageDownloadSizeBytes {
		downloaded = runtime.ImageDownloadSizeBytes
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"variant_id":                runtime.VariantID,
		"container":                 runtime.Container,
		"state":                     state,
		"image":                     runtime.Image,
		"image_download_size_bytes": runtime.ImageDownloadSizeBytes,
		"image_downloaded_bytes":    downloaded,
	})
}

func (c *controller) deleteModule(w http.ResponseWriter, r *http.Request) {
	runtime, ok := c.byVariant[r.PathValue("variant")]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "runtime variant is not allowlisted"})
		return
	}
	inspected, found, err := c.inspectOwned(r.Context(), runtime.Container)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if !found {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if !c.matches(inspected, runtime) {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "container belongs to a different runtime"})
		return
	}
	if inspected.State.Running {
		writeJSON(w, http.StatusConflict, map[string]string{"error": "active module cannot be removed"})
		return
	}
	status, body, err := c.dockerRequest(r.Context(), http.MethodDelete, "/containers/"+url.PathEscape(runtime.Container), nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	if status == http.StatusNoContent || status == http.StatusNotFound {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	writeDockerError(w, status, body)
}

func (c *controller) ensureModule(ctx context.Context, runtime runtimePolicy) error {
	inspected, found, err := c.inspectOwned(ctx, runtime.Container)
	if err != nil {
		return err
	}
	if found {
		if !c.owned(inspected) {
			return fmt.Errorf("refusing to replace unowned container %q", runtime.Container)
		}
		if c.matches(inspected, runtime) {
			return nil
		}
		if inspected.State.Running {
			return fmt.Errorf("runtime container %q is active with a different policy", runtime.Container)
		}
		status, body, err := c.dockerRequest(ctx, http.MethodDelete, "/containers/"+url.PathEscape(runtime.Container), nil)
		if err != nil || (status != http.StatusNoContent && status != http.StatusNotFound) {
			return fmt.Errorf("remove stale runtime container: %s: %w", strings.TrimSpace(string(body)), err)
		}
	}
	available, err := c.imageAvailable(ctx, runtime.Image)
	if err != nil {
		return err
	}
	if !available {
		if err := c.pullImage(ctx, runtime.Image); err != nil {
			return err
		}
	}
	env := make([]string, 0, len(runtime.Environment))
	keys := make([]string, 0, len(runtime.Environment))
	for key := range runtime.Environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		env = append(env, key+"="+runtime.Environment[key])
	}
	labels := map[string]string{
		"aurago.managed":     ownerLabel,
		"aurago.component":   "speech-lab",
		"aurago.role":        "module",
		"aurago.bundle":      c.policy.BundleVersion,
		"aurago.fingerprint": c.bundleDigest,
		"s2s.lab.managed":    "true",
		"stage":              runtime.Stage,
		"backend-id":         runtime.BackendID,
		"variant-id":         runtime.VariantID,
		"s2s.image":          runtime.Image,
	}
	binds := []string{}
	mounts := runtime.ModelMounts
	if len(mounts) == 0 {
		mounts = []string{"/models"}
	}
	for _, target := range mounts {
		if target != "/models" && target != "/opt/s2s-models" {
			return fmt.Errorf("runtime %q has unsafe model mount %q", runtime.VariantID, target)
		}
		binds = append(binds, c.modelsVolume+":"+target+":ro")
	}
	binds = append(binds, c.dataVolume+":/data")
	hostConfig := map[string]any{
		"NetworkMode":    c.network,
		"RestartPolicy":  map[string]any{"Name": "no"},
		"Binds":          binds,
		"Tmpfs":          map[string]string{"/tmp": "rw,nosuid,nodev,exec,size=268435456"},
		"ReadonlyRootfs": true,
		"CapDrop":        []string{"ALL"},
		"CapAdd":         []string{"SETUID", "SETGID"},
		"SecurityOpt":    []string{"no-new-privileges:true"},
	}
	if runtime.ShmSizeBytes > 0 {
		hostConfig["ShmSize"] = runtime.ShmSizeBytes
	} else if runtime.Resources.ShmBytes > 0 {
		hostConfig["ShmSize"] = runtime.Resources.ShmBytes
	}
	if runtime.Resources.MemoryBytes > 0 {
		hostConfig["Memory"] = runtime.Resources.MemoryBytes
	}
	if runtime.Resources.NanoCPUs > 0 {
		hostConfig["NanoCpus"] = runtime.Resources.NanoCPUs
	}
	if runtime.Accelerator == "cuda" {
		hostConfig["DeviceRequests"] = []map[string]any{{"Driver": "nvidia", "Count": -1, "Capabilities": [][]string{{"gpu"}}}}
	}
	if runtime.Accelerator == "vulkan" || runtime.Accelerator == "sycl" {
		hostConfig["Devices"] = []map[string]string{{"PathOnHost": "/dev/dri", "PathInContainer": "/dev/dri", "CgroupPermissions": "rwm"}}
	}
	aliases := append([]string{runtime.Container}, runtime.Aliases...)
	bodyMap := map[string]any{
		"Image":            runtime.Image,
		"Cmd":              runtime.Command,
		"Env":              env,
		"Labels":           labels,
		"HostConfig":       hostConfig,
		"NetworkingConfig": map[string]any{"EndpointsConfig": map[string]any{c.network: map[string]any{"Aliases": aliases}}},
	}
	bodyJSON, _ := json.Marshal(bodyMap)
	status, body, err := c.dockerRequest(ctx, http.MethodPost, "/containers/create?name="+url.QueryEscape(runtime.Container), bodyJSON)
	if err != nil {
		return fmt.Errorf("create runtime container: %w", err)
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("create runtime container returned HTTP %d: %s", status, strings.TrimSpace(string(body)))
	}
	return nil
}

func (c *controller) imageAvailable(ctx context.Context, image string) (bool, error) {
	status, body, err := c.dockerRequest(ctx, http.MethodGet, "/images/"+url.PathEscape(image)+"/json", nil)
	if err != nil {
		return false, fmt.Errorf("inspect runtime image: %w", err)
	}
	switch status {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("inspect runtime image returned HTTP %d: %s", status, strings.TrimSpace(string(body)))
	}
}

func (c *controller) inspectOwned(ctx context.Context, name string) (dockerInspect, bool, error) {
	var inspected dockerInspect
	if _, ok := c.byContainer[name]; !ok {
		return inspected, false, fmt.Errorf("container %q is not allowlisted", name)
	}
	status, body, err := c.dockerRequest(ctx, http.MethodGet, "/containers/"+url.PathEscape(name)+"/json", nil)
	if err != nil {
		return inspected, false, err
	}
	if status == http.StatusNotFound {
		return inspected, false, nil
	}
	if status < 200 || status >= 300 {
		return inspected, false, fmt.Errorf("Docker inspect returned HTTP %d", status)
	}
	if err := json.Unmarshal(body, &inspected); err != nil {
		return inspected, false, fmt.Errorf("parse Docker inspect: %w", err)
	}
	return inspected, true, nil
}

func (c *controller) owned(inspected dockerInspect) bool {
	labels := inspected.Config.Labels
	return labels["aurago.managed"] == ownerLabel && labels["s2s.lab.managed"] == "true" && labels["aurago.role"] == "module"
}

func (c *controller) matches(inspected dockerInspect, runtime runtimePolicy) bool {
	labels := inspected.Config.Labels
	return c.owned(inspected) &&
		labels["aurago.bundle"] == c.policy.BundleVersion &&
		labels["aurago.fingerprint"] == c.bundleDigest &&
		labels["stage"] == runtime.Stage &&
		labels["backend-id"] == runtime.BackendID &&
		labels["variant-id"] == runtime.VariantID &&
		labels["s2s.image"] == runtime.Image &&
		inspected.Config.Image == runtime.Image
}

func (c *controller) matchesAny(inspected dockerInspect) bool {
	name := strings.TrimPrefix(inspected.Name, "/")
	for _, runtime := range c.byContainer[name] {
		if c.matches(inspected, runtime) {
			return true
		}
	}
	return false
}

func (c *controller) matchesListed(name, image string, labels map[string]any) bool {
	for _, runtime := range c.byContainer[name] {
		if labelString(labels, "aurago.managed") == ownerLabel &&
			labelString(labels, "s2s.lab.managed") == "true" &&
			labelString(labels, "aurago.role") == "module" &&
			labelString(labels, "aurago.bundle") == c.policy.BundleVersion &&
			labelString(labels, "aurago.fingerprint") == c.bundleDigest &&
			labelString(labels, "stage") == runtime.Stage &&
			labelString(labels, "backend-id") == runtime.BackendID &&
			labelString(labels, "variant-id") == runtime.VariantID &&
			labelString(labels, "s2s.image") == runtime.Image && image == runtime.Image {
			return true
		}
	}
	return false
}

func (c *controller) dockerRequest(ctx context.Context, method, path string, body []byte) (int, []byte, error) {
	var reader io.Reader
	if body != nil {
		reader = strings.NewReader(string(body))
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://docker/"+dockerAPIVer+path, reader)
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.docker.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	return resp.StatusCode, data, err
}

func (c *controller) pullImage(ctx context.Context, image string) error {
	c.pullMu.Lock()
	if c.imagePulls == nil {
		c.imagePulls = make(map[string]int64)
	}
	c.imagePulls[image] = 0
	c.pullMu.Unlock()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://docker/"+dockerAPIVer+"/images/create?fromImage="+url.QueryEscape(image), nil)
	if err != nil {
		return fmt.Errorf("prepare runtime image pull: %w", err)
	}
	resp, err := c.docker.Do(req)
	if err != nil {
		return fmt.Errorf("pull runtime image: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<10))
		return fmt.Errorf("pull runtime image returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	// Docker sends a stream of JSON events. Read it to EOF so a long pull is
	// not silently treated as complete after the ordinary response-size cap.
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	layers := make(map[string]dockerLayerProgress)
	for scanner.Scan() {
		if len(scanner.Bytes()) == 0 {
			continue
		}
		var event struct {
			Error          string `json:"error"`
			ID             string `json:"id"`
			Status         string `json:"status"`
			ProgressDetail struct {
				Current int64 `json:"current"`
				Total   int64 `json:"total"`
			} `json:"progressDetail"`
			ErrorDetail struct {
				Message string `json:"message"`
			} `json:"errorDetail"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return fmt.Errorf("decode runtime image pull: %w", err)
		}
		message := strings.TrimSpace(event.Error)
		if message == "" {
			message = strings.TrimSpace(event.ErrorDetail.Message)
		}
		if message != "" {
			return fmt.Errorf("pull runtime image: %s", message)
		}
		if downloaded := updateDockerLayerProgress(layers, event.ID, event.Status, event.ProgressDetail.Current, event.ProgressDetail.Total); downloaded >= 0 {
			c.pullMu.Lock()
			if downloaded > c.imagePulls[image] {
				c.imagePulls[image] = downloaded
			}
			c.pullMu.Unlock()
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read runtime image pull: %w", err)
	}
	return nil
}

type dockerLayerProgress struct {
	current int64
	total   int64
}

func updateDockerLayerProgress(layers map[string]dockerLayerProgress, id, status string, current, total int64) int64 {
	if id == "" {
		return -1
	}
	state := layers[id]
	if strings.EqualFold(status, "Download complete") || strings.EqualFold(status, "Pull complete") {
		state.current = state.total
	} else if strings.EqualFold(status, "Downloading") && total > 0 {
		if current > state.current {
			state.current = current
		}
		if total > state.total {
			state.total = total
		}
	} else {
		return -1
	}
	layers[id] = state
	var downloaded int64
	for _, layer := range layers {
		downloaded += layer.current
	}
	return downloaded
}

func writeDockerError(w http.ResponseWriter, status int, body []byte) {
	message := strings.TrimSpace(string(body))
	if message == "" {
		message = http.StatusText(status)
	}
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func labelString(labels map[string]any, key string) string {
	value, _ := labels[key].(string)
	return value
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
