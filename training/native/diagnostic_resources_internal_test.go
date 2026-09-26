package native

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

const gpuQuery = "index,name,compute_cap,memory.total,memory.free"

// ProbeGPUs asks the driver what this host has.
//
// It refuses rather than reporting zero cards. A host that answers "no GPU" and
// a host whose driver could not be reached look identical to a caller that gets
// an empty list, and the second one is a broken node reporting itself healthy.
func diagnosticProbeGPUs(ctx context.Context, fraction float64) (GPUProbe, error) {
	if fraction <= 0 || fraction > 1 {
		return GPUProbe{}, fmt.Errorf("cluster: the VRAM fraction has to be in (0, 1]; got %v", fraction)
	}
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu="+gpuQuery, "--format=csv,noheader,nounits").Output()
	if err != nil {
		return GPUProbe{}, fmt.Errorf("cluster: reading the driver with nvidia-smi: %w", err)
	}
	gpus, err := parseGPUs(string(out), fraction)
	if err != nil {
		return GPUProbe{}, err
	}
	if len(gpus) == 0 {
		return GPUProbe{}, fmt.Errorf("cluster: the driver reported no card; check NVIDIA_VISIBLE_DEVICES")
	}
	driver, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=driver_version", "--format=csv,noheader").Output()
	if err != nil {
		return GPUProbe{}, fmt.Errorf("cluster: reading the driver version: %w", err)
	}
	version := strings.TrimSpace(string(driver))
	if i := strings.IndexByte(version, '\n'); i >= 0 {
		version = version[:i]
	}
	return GPUProbe{Driver: strings.TrimSpace(version), VRAMFraction: fraction, GPUs: gpus}, nil
}

// parseGPUs reads the CSV the driver printed. It is separate from ProbeGPUs so a
// test can exercise it without a card.
func parseGPUs(out string, fraction float64) ([]GPU, error) {
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) != 5 {
			return nil, fmt.Errorf("cluster: the driver printed %d columns for %q, want 5", len(fields), strings.TrimSpace(line))
		}
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		index, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, fmt.Errorf("cluster: the driver printed %q as a card index: %w", fields[0], err)
		}
		total, err := mib(fields[3])
		if err != nil {
			return nil, err
		}
		free, err := mib(fields[4])
		if err != nil {
			return nil, err
		}
		gpus = append(gpus, GPU{
			Index: index, Name: fields[1], Capability: fields[2],
			TotalBytes: total, FreeBytes: free,
			CapBytes: int64(float64(total) * fraction),
		})
	}
	return gpus, nil
}

// mib converts the driver's mebibytes to bytes.
func mib(s string) (int64, error) {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("cluster: the driver printed %q where a size in MiB was expected: %w", s, err)
	}
	return v << 20, nil
}

// ReadHostAvailableRAM reads Linux /proc/meminfo. The kernel labels the field
// kB but defines it in binary KiB, so the conversion is value * 1024 bytes.
func diagnosticHostAvailableRAM(path string) (int64, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("reading host RAM metric: %w", err)
	}
	for _, line := range strings.Split(string(body), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 3 || fields[0] != "MemAvailable:" {
			continue
		}
		if fields[2] != "kB" {
			return 0, errors.New("MemAvailable has an unexpected unit")
		}
		kib, parseErr := strconv.ParseInt(fields[1], 10, 64)
		if parseErr != nil || kib < 1 || kib > (1<<63-1)/1024 {
			return 0, errors.New("MemAvailable has an invalid value")
		}
		return kib * 1024, nil
	}
	return 0, errors.New("MemAvailable is absent from host metrics")
}

func experimentNativeCgroupAvailable() (int64, error) {
	maximum, err := os.ReadFile("/sys/fs/cgroup/memory.max")
	if err != nil {
		return 0, errors.New("experiment native: cgroup v2 memory limit unavailable")
	}
	if strings.TrimSpace(string(maximum)) == "max" {
		return math.MaxInt64, nil
	}
	limit, err := strconv.ParseInt(strings.TrimSpace(string(maximum)), 10, 64)
	if err != nil || limit <= 0 {
		return 0, errors.New("experiment native: invalid cgroup memory limit")
	}
	current, err := os.ReadFile("/sys/fs/cgroup/memory.current")
	if err != nil {
		return 0, errors.New("experiment native: cgroup memory usage unavailable")
	}
	used, err := strconv.ParseInt(strings.TrimSpace(string(current)), 10, 64)
	if err != nil || used < 0 || used > limit {
		return 0, errors.New("experiment native: invalid cgroup memory usage")
	}
	return limit - used, nil
}
