MX_REPOSITORY := https://github.com/mxxm-t/mx-llama.cpp
MX_REVISION := a245214d8df6304762c7688c6b8ee45652c5c8e5
MX_SOURCE ?= $(CURDIR)/source
MX_BUILD ?= $(CURDIR)/build-sm75-64

.PHONY: source configure build

source:
	test -d "$(MX_SOURCE)/.git" || git clone "$(MX_REPOSITORY)" "$(MX_SOURCE)"
	git -C "$(MX_SOURCE)" fetch --depth 1 origin "$(MX_REVISION)"
	git -C "$(MX_SOURCE)" checkout --detach "$(MX_REVISION)"

configure: source
	cmake -S "$(MX_SOURCE)" -B "$(MX_BUILD)" \
		-DGGML_CUDA=ON \
		-DGGML_RPC=ON \
		-DGGML_CUDA_NCCL=ON \
		-DGGML_NATIVE=OFF \
		-DLLAMA_CURL=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=75 \
		-DCMAKE_C_FLAGS=-DGGML_SCHED_MAX_BACKENDS=64 \
		-DCMAKE_CXX_FLAGS=-DGGML_SCHED_MAX_BACKENDS=64

build: configure
	cmake --build "$(MX_BUILD)" --config Release --parallel
