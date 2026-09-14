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
	mxRevision          = "a245214d8df6304762c7688c6b8ee45652c5c8e5"
	mxRPCDigest         = "182a313ddef3c90703a6d38cd90dc66bfccf4aeabbc93cdd85494eabf0332dd5"
	mxCLIDigest         = "604a5ad57e3545e1ff4e119bc78a280a98075a59c228305e21dce41c38791fb9"
	mxServerDigest      = "482302ee1b9f9145da85c753231399de423021213b071d32eb75cf8fba88f438"
	mxRuntimeDigest     = "8e3296f980b6a01e3933ce6fba2e77c933df112a0bbc419acb4a22b724a8c8d0"
	mxManifestDigest    = "919222c9259f179a890c98b549638ebdd17a84ce06ef02bb3650b8c1bdc7193e"
	mxModelRelativePath = "DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf"
)

// MXManifest identifies the source, build and executables admitted by this backend.
type MXManifest struct {
	Backend           string
	Repository        string
	Revision          string
	CUDAArchitecture  string
	SchedulerBackends int
	RuntimeDigest     string
	ModelDigest       string
	Binaries          map[string]string
}

// AdmittedMXManifest returns the immutable identity accepted by this release.
func AdmittedMXManifest() MXManifest {
	return MXManifest{
		Backend: MXBackendID, Repository: "https://github.com/mxxm-t/mx-llama.cpp",
		Revision: mxRevision, CUDAArchitecture: "75", SchedulerBackends: 64,
		RuntimeDigest: mxRuntimeDigest, ModelDigest: mxManifestDigest,
		Binaries: map[string]string{
			"llama-cli": mxCLIDigest, "llama-server": mxServerDigest,
			"ggml-rpc-server": mxRPCDigest,
		},
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
}

// MXConfig names the installed, versioned runtime and model cache.
type MXConfig struct {
	RuntimeRoot string
	ModelRoot   string
}

// MXBackend resolves and verifies one installed mx-llama.cpp release.
type MXBackend struct {
	runtimeRoot string
	modelRoot   string
}

// NewMXBackend verifies the runtime identity before returning a backend.
func NewMXBackend(cfg MXConfig) (*MXBackend, error) {
	if !filepath.IsAbs(cfg.RuntimeRoot) || !filepath.IsAbs(cfg.ModelRoot) {
		return nil, errors.New("llama: MX runtime and model roots have to be absolute")
	}
	b := &MXBackend{runtimeRoot: filepath.Clean(cfg.RuntimeRoot), modelRoot: filepath.Clean(cfg.ModelRoot)}
	for name, want := range AdmittedMXManifest().Binaries {
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

// Generate builds the coordinator process without accepting an executable or model path.
func (b *MXBackend) Generate(in MXGenerateRequest) (NativeProcess, error) {
	if err := validateMXEndpoints(in.RPCEndpoints); err != nil {
		return NativeProcess{}, err
	}
	if strings.TrimSpace(in.Prompt) == "" || len(in.Prompt) > 4096 {
		return NativeProcess{}, errors.New("llama: MX prompt has to contain between 1 and 4096 bytes")
	}
	if in.MaxTokens < 1 || in.MaxTokens > 256 {
		return NativeProcess{}, errors.New("llama: MX max tokens has to be between 1 and 256")
	}
	if in.Context != 4096 {
		return NativeProcess{}, errors.New("llama: MX qualification context has to be 4096")
	}
	if in.Deadline < time.Minute || in.Deadline > 3*time.Hour {
		return NativeProcess{}, errors.New("llama: MX generation deadline has to be between one minute and three hours")
	}
	if err := verifyFile(filepath.Join(b.modelRoot, "SHA256SUMS"), mxManifestDigest); err != nil {
		return NativeProcess{}, fmt.Errorf("llama: verify MX model manifest: %w", err)
	}
	model := filepath.Join(b.modelRoot, mxModelRelativePath)
	if info, err := os.Stat(model); err != nil || !info.Mode().IsRegular() {
		return NativeProcess{}, errors.New("llama: admitted MX model shard is absent")
	}
	args := []string{"--signal=TERM", "--kill-after=60s", strconv.FormatInt(int64(in.Deadline.Seconds()), 10) + "s",
		filepath.Join(b.runtimeRoot, "bin", "llama-cli"),
		"-m", model, "--rpc", strings.Join(in.RPCEndpoints, ","), "--split-mode", "layer",
		"--gpu-layers", "auto", "--fit", "on", "--fit-target", "3200", "--fit-ctx", "4096",
		"-c", "4096", "-b", "512", "-ub", "128", "--load-mode", "dio", "--lazy-mode", "auto",
		"--no-warmup", "--seed", "59", "--temp", "0", "-n", strconv.Itoa(in.MaxTokens),
		"--no-display-prompt", "-p", in.Prompt,
	}
	return NativeProcess{Path: "/usr/bin/timeout", Args: args}, nil
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
