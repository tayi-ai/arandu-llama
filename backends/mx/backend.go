// Package mx exposes the qualified mx-llama.cpp subprocess backend without
// linking the in-process llama.cpp engine.
package mx

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// MXBackendID is the stable selector for the subprocess backend.
const MXBackendID = "mx-llama.cpp"

const (
	mxRevision       = "a245214d8df6304762c7688c6b8ee45652c5c8e5"
	mxRPCDigest      = "182a313ddef3c90703a6d38cd90dc66bfccf4aeabbc93cdd85494eabf0332dd5"
	mxCLIDigest      = "604a5ad57e3545e1ff4e119bc78a280a98075a59c228305e21dce41c38791fb9"
	mxServerDigest   = "482302bc25ca9b2ee5b1cc1532508ab9ea62d22f841caa9846eadd576f72eccc"
	mxLibLlamaDigest = "92cb8adca8177feb417a9c3aee856ee1a72f1390f387459b9a533fd52c0408ad"
	mxRuntimeDigest  = "77c0d31dcf59a4798289f90279f3e25668873edd059d7701b47b9d8a10d4a05d"
)

// MXModelRecipe identifies an installation-owned model representation.
// The zero value is not admitted.
type MXModelRecipe string

// MXModelIdentity pins model provenance and the installed shard manifest.
// Applications provide it through their validated configuration, never from
// an execution request. Runtime qualification is a separate identity.
type MXModelIdentity struct {
	Recipe       MXModelRecipe
	Repository   string
	Revision     string
	Quantisation string
	Manifest     string
	FirstShard   string
	Shards       int
}

// MXManifest identifies the source, build and executables admitted by this backend.
type MXManifest struct {
	Backend            string
	Repository         string
	Revision           string
	CUDAArchitecture   string
	SchedulerBackends  int
	RuntimeDigest      string
	ModelDigest        string
	ModelRecipe        MXModelRecipe
	ModelRepository    string
	ModelRevision      string
	ModelQuantisation  string
	ModelFirstShard    string
	ModelShards        int
	Binaries           map[string]string
	RuntimeEnvironment map[string]string
}

func runtimeManifest(model MXModelIdentity) MXManifest {
	return MXManifest{
		Backend: MXBackendID, Repository: "https://github.com/mxxm-t/mx-llama.cpp",
		Revision: mxRevision, CUDAArchitecture: "75", SchedulerBackends: 64,
		RuntimeDigest: mxRuntimeDigest, ModelDigest: model.Manifest,
		ModelRecipe: model.Recipe, ModelRepository: model.Repository,
		ModelRevision: model.Revision, ModelQuantisation: model.Quantisation,
		ModelFirstShard: model.FirstShard, ModelShards: model.Shards,
		Binaries: map[string]string{
			"llama-cli": mxCLIDigest, "llama-server": mxServerDigest,
			"libllama.so.0.3.0": mxLibLlamaDigest,
			"ggml-rpc-server":   mxRPCDigest,
		},
		RuntimeEnvironment: map[string]string{"GGML_CUDA_Q8_1_CACHE": "0"},
	}
}

// NativeProcess is one typed native invocation.
//
// It is a path and arguments rather than a shell command. The caller may hand
// it to its process supervisor without reparsing text or accepting executable
// paths from a request.
type NativeProcess struct {
	Path string
	Args []string
	Env  []string
}

// MXConfig names the installed, versioned runtime and model cache.
type MXConfig struct {
	RuntimeRoot string
	ModelRoot   string
	Model       MXModelIdentity
}

// MXBackend resolves and verifies one installed mx-llama.cpp release.
type MXBackend struct {
	runtimeRoot string
	modelRoot   string
	model       MXModelIdentity
}

// NewMXBackend verifies the runtime identity before returning a backend.
func NewMXBackend(cfg MXConfig) (*MXBackend, error) {
	if !filepath.IsAbs(cfg.RuntimeRoot) || !filepath.IsAbs(cfg.ModelRoot) {
		return nil, errors.New("llama: MX runtime and model roots have to be absolute")
	}
	if err := validateModel(cfg.Model); err != nil {
		return nil, err
	}
	manifest := runtimeManifest(cfg.Model)
	b := &MXBackend{runtimeRoot: filepath.Clean(cfg.RuntimeRoot), modelRoot: filepath.Clean(cfg.ModelRoot), model: cfg.Model}
	for name, want := range manifest.Binaries {
		if err := verifyFile(filepath.Join(b.runtimeRoot, "bin", name), want); err != nil {
			return nil, fmt.Errorf("llama: verify MX %s: %w", name, err)
		}
	}
	return b, nil
}

// MXRPCRequest is the bounded RPC process one GPU participant may expose.
type MXRPCRequest struct {
	BindIP   string
	Port     int
	Devices  []int
	Deadline time.Duration
}

// RPC builds the admitted RPC server process.
func (b *MXBackend) RPC(in MXRPCRequest) (NativeProcess, error) {
	ip := net.ParseIP(in.BindIP)
	if ip == nil || ip.To4() == nil || !privateOrCGNATAddress(ip) {
		return NativeProcess{}, errors.New("llama: MX RPC bind address has to be private or CGNAT IPv4")
	}
	if in.Port != 50052 {
		return NativeProcess{}, errors.New("llama: MX RPC port is not admitted")
	}
	if len(in.Devices) != 2 || in.Devices[0] != 0 || in.Devices[1] != 1 {
		return NativeProcess{}, errors.New("llama: MX RPC requires the measured CUDA0,CUDA1 pair")
	}
	if in.Deadline < time.Minute || in.Deadline > 3*time.Hour {
		return NativeProcess{}, errors.New("llama: MX RPC deadline has to be between one minute and three hours")
	}
	return NativeProcess{
		Path: "/usr/bin/timeout",
		Args: []string{"--signal=TERM", "--kill-after=30s", strconv.FormatInt(int64(in.Deadline.Seconds()), 10) + "s",
			filepath.Join(b.runtimeRoot, "bin", "ggml-rpc-server"), "-H", in.BindIP, "-p", strconv.Itoa(in.Port), "-d", "CUDA0,CUDA1", "-c"},
		Env: mxRuntimeEnvironment(),
	}, nil
}

// MXGenerateRequest is the bounded coordinator operation admitted for the smoke.
type MXGenerateRequest struct {
	RPCEndpoints []string
	Prompt       string
	MaxTokens    int
	Context      int
	Deadline     time.Duration
}

// MXServeRequest is the resident, authenticated coordinator admitted for
// repeated scoring and generation. RPC participants keep the model distributed
// while the HTTP process owns the local context on coordinator one.
type MXServeRequest struct {
	RPCEndpoints []string
	BindIP       string
	Port         int
	Context      int
	Deadline     time.Duration
	Adapters     []MXAdapter
}

// MXAdapter identifies one immutable GGUF LoRA available to the coordinator.
// It is loaded at scale zero so each measurement can select exactly one
// complete factor-space candidate without restarting the quantized base.
type MXAdapter struct {
	Path   string
	SHA256 string
}

// Serve builds a resident llama-server without accepting an executable or
// model path. Authentication is supplied by the supervising application in the
// LLAMA_API_KEY environment variable, so it never appears in process arguments.
func (b *MXBackend) Serve(in MXServeRequest) (NativeProcess, error) {
	if err := validateMXEndpoints(in.RPCEndpoints); err != nil {
		return NativeProcess{}, err
	}
	ip := net.ParseIP(in.BindIP)
	if ip == nil || ip.To4() == nil || !privateOrCGNATAddress(ip) {
		return NativeProcess{}, errors.New("llama: MX server bind address has to be private or CGNAT IPv4")
	}
	if in.Port != 50053 {
		return NativeProcess{}, errors.New("llama: MX server port is not admitted")
	}
	if in.Context != 4096 {
		return NativeProcess{}, errors.New("llama: MX server qualification context has to be 4096")
	}
	if in.Deadline < time.Minute || in.Deadline > 12*time.Hour {
		return NativeProcess{}, errors.New("llama: MX server deadline has to be between one minute and twelve hours")
	}
	if len(in.Adapters) > 16 {
		return NativeProcess{}, errors.New("llama: MX server admits at most 16 factor-space candidates")
	}
	seenAdapters := make(map[string]bool, len(in.Adapters))
	for _, adapter := range in.Adapters {
		if !filepath.IsAbs(adapter.Path) || strings.ContainsAny(adapter.Path, ",:") {
			return NativeProcess{}, errors.New("llama: MX adapter path has to be absolute and contain no list separators")
		}
		if seenAdapters[adapter.Path] {
			return NativeProcess{}, errors.New("llama: MX adapter paths have to be unique")
		}
		seenAdapters[adapter.Path] = true
		if len(adapter.SHA256) != 64 {
			return NativeProcess{}, errors.New("llama: MX adapter digest is malformed")
		}
		if err := verifyFile(adapter.Path, adapter.SHA256); err != nil {
			return NativeProcess{}, fmt.Errorf("llama: verify MX adapter %s: %w", filepath.Base(adapter.Path), err)
		}
	}
	model, err := b.admittedModel()
	if err != nil {
		return NativeProcess{}, err
	}
	return b.serveProcess(model, in), nil
}

func (b *MXBackend) serveProcess(model string, in MXServeRequest) NativeProcess {
	args := []string{"--signal=TERM", "--kill-after=60s", strconv.FormatInt(int64(in.Deadline.Seconds()), 10) + "s",
		filepath.Join(b.runtimeRoot, "bin", "llama-server"),
		"-m", model, "--rpc", strings.Join(in.RPCEndpoints, ","), "--split-mode", "layer",
		"--gpu-layers", "auto", "--fit", "on", "--fit-target", "3200", "--fit-ctx", "4096",
		"-c", "4096", "-b", "512", "-ub", "128", "--load-mode", "dio", "--lazy-mode", "auto",
		"--no-host", "--no-repack", "--no-warmup", "--host", in.BindIP, "--port", strconv.Itoa(in.Port),
		"--alias", string(b.model.Recipe), "--metrics",
	}
	if len(in.Adapters) > 0 {
		adapters := make([]string, 0, len(in.Adapters))
		for _, adapter := range in.Adapters {
			adapters = append(adapters, adapter.Path+":0")
		}
		args = append(args, "--lora-scaled", strings.Join(adapters, ","))
	}
	return NativeProcess{Path: "/usr/bin/timeout", Args: args, Env: mxRuntimeEnvironment()}
}

// Generate builds the coordinator process without accepting an executable or model path.
func (b *MXBackend) Generate(in MXGenerateRequest) (NativeProcess, error) {
	if err := validateMXEndpoints(in.RPCEndpoints); err != nil {
		return NativeProcess{}, err
	}
	if strings.TrimSpace(in.Prompt) == "" || len(in.Prompt) > 4096 {
		return NativeProcess{}, errors.New("llama: MX prompt has to contain between 1 and 4096 bytes")
	}
	if in.MaxTokens < 1 || in.MaxTokens > 1024 {
		return NativeProcess{}, errors.New("llama: MX max tokens has to be between 1 and 1024")
	}
	if in.Context != 4096 {
		return NativeProcess{}, errors.New("llama: MX qualification context has to be 4096")
	}
	if in.Deadline < time.Minute || in.Deadline > 3*time.Hour {
		return NativeProcess{}, errors.New("llama: MX generation deadline has to be between one minute and three hours")
	}
	model, err := b.admittedModel()
	if err != nil {
		return NativeProcess{}, err
	}
	return b.generateProcess(model, in), nil
}

func (b *MXBackend) generateProcess(model string, in MXGenerateRequest) NativeProcess {
	args := []string{"--signal=TERM", "--kill-after=60s", strconv.FormatInt(int64(in.Deadline.Seconds()), 10) + "s",
		filepath.Join(b.runtimeRoot, "bin", "llama-cli"),
		"-m", model, "--rpc", strings.Join(in.RPCEndpoints, ","), "--split-mode", "layer",
		"--gpu-layers", "auto", "--fit", "on", "--fit-target", "3200", "--fit-ctx", "4096",
		"-c", "4096", "-b", "512", "-ub", "128", "--load-mode", "dio", "--lazy-mode", "auto",
		"--no-host", "--no-repack", "--no-warmup", "--single-turn", "--seed", "59", "--temp", "0", "-n", strconv.Itoa(in.MaxTokens),
		"--no-display-prompt", "-p", in.Prompt,
	}
	return NativeProcess{Path: "/usr/bin/timeout", Args: args, Env: mxRuntimeEnvironment()}
}

func (b *MXBackend) admittedModel() (string, error) {
	if err := verifyFile(filepath.Join(b.modelRoot, "SHA256SUMS"), b.model.Manifest); err != nil {
		return "", fmt.Errorf("llama: verify MX model manifest for %s: %w", b.model.Recipe, err)
	}
	model := filepath.Join(b.modelRoot, b.model.FirstShard)
	if info, err := os.Stat(model); err != nil || !info.Mode().IsRegular() {
		return "", errors.New("llama: admitted MX model shard is absent")
	}
	return model, nil
}

// AdmittedModelPath verifies and returns the first shard of the pinned model.
// Callers may inspect its GGUF metadata but cannot select another model path.
func (b *MXBackend) AdmittedModelPath() (string, error) { return b.admittedModel() }

// mxRuntimeEnvironment fixes every native process to the qualified runtime
// behavior. The pinned fork documents that disabling this optional cache
// restores its previous CUDA path. A duplicate inherited value must not be able
// to reactivate the cache after the release was admitted.
func mxRuntimeEnvironment() []string {
	const key = "GGML_CUDA_Q8_1_CACHE="
	environment := make([]string, 0, len(os.Environ())+1)
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, key) {
			environment = append(environment, value)
		}
	}
	return append(environment, key+"0")
}

func validateMXEndpoints(endpoints []string) error {
	if len(endpoints) != 20 {
		return fmt.Errorf("llama: MX generation needs 20 RPC endpoints; got %d", len(endpoints))
	}
	seen := make(map[string]bool, len(endpoints))
	for _, endpoint := range endpoints {
		host, port, err := net.SplitHostPort(endpoint)
		if err != nil || port != "50052" {
			return errors.New("llama: MX RPC endpoint is malformed or uses an unadmitted port")
		}
		ip := net.ParseIP(host)
		if ip == nil || ip.To4() == nil || !privateOrCGNATAddress(ip) {
			return errors.New("llama: MX RPC endpoint has to use private or CGNAT IPv4")
		}
		if seen[endpoint] {
			return errors.New("llama: MX RPC endpoints have to be unique")
		}
		seen[endpoint] = true
	}
	return nil
}

func privateOrCGNATAddress(ip net.IP) bool {
	_, cgnat, _ := net.ParseCIDR("100.64.0.0/10")
	return ip.IsPrivate() || cgnat.Contains(ip)
}

func verifyFile(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("sha256 is %s, want %s", got, want)
	}
	return nil
}
