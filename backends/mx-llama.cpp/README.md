# Backend mx-llama.cpp

Este backend adiciona a execução qualificada do DeepSeek-V4.1-Flash sem
substituir o backend `llama.cpp` já integrado ao módulo. O código Go nunca
recebe uma linha de shell, um caminho de executável ou um caminho de modelo da
requisição. Ele produz um `NativeProcess` tipado depois de validar a topologia,
os limites e os arquivos instalados.

## Identidade da release

- upstream: `https://github.com/mxxm-t/mx-llama.cpp`
- revisão: `a245214d8df6304762c7688c6b8ee45652c5c8e5`
- arquitetura CUDA: SM75
- limite do scheduler: `GGML_SCHED_MAX_BACKENDS=64`
- backends: CUDA, RPC e NCCL
- modo nativo: desligado

O `runtime_digest` é o SHA-256 da sequência abaixo, com uma quebra de linha
depois de cada item, inclusive o último:

```text
a245214d8df6304762c7688c6b8ee45652c5c8e5
75
64
llama-cli:604a5ad57e3545e1ff4e119bc78a280a98075a59c228305e21dce41c38791fb9
llama-server:482302ee1b9f9145da85c753231399de423021213b071d32eb75cf8fba88f438
ggml-rpc-server:182a313ddef3c90703a6d38cd90dc66bfccf4aeabbc93cdd85494eabf0332dd5
```

O resultado qualificado é
`8e3296f980b6a01e3933ce6fba2e77c933df112a0bbc419acb4a22b724a8c8d0`.
O arquivo `manifest.json`, a constante Go e o manifesto instalado precisam
concordar antes de uma release ser admitida.

## Construção reproduzível

`make build` busca somente a revisão fixada e cria `build-sm75-64`. Depois da
construção, compare os três binários com `manifest.json`. Uma divergência cria
uma nova release qualificada; ela não deve ser corrigida alterando o hash no
host.

O modelo é identificado separadamente pelo SHA-256 do seu `SHA256SUMS`.
Assim, trocar pesos não se disfarça de recompilação do runtime, e recompilar o
runtime não se disfarça de troca de modelo.
