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
patch:max-devices-64:5a2e22b2a1a1bf3009bb775b31bf0e9ed65474cc2912387a00b9076d406676cc
llama-cli:604a5ad57e3545e1ff4e119bc78a280a98075a59c228305e21dce41c38791fb9
llama-server:482302bc25ca9b2ee5b1cc1532508ab9ea62d22f841caa9846eadd576f72eccc
ggml-rpc-server:182a313ddef3c90703a6d38cd90dc66bfccf4aeabbc93cdd85494eabf0332dd5
libllama.so.0.3.0:92cb8adca8177feb417a9c3aee856ee1a72f1390f387459b9a533fd52c0408ad
env:GGML_CUDA_Q8_1_CACHE=0
```

O resultado qualificado é
`77c0d31dcf59a4798289f90279f3e25668873edd059d7701b47b9d8a10d4a05d`.
O arquivo `manifest.json`, a constante Go e o manifesto instalado precisam
concordar antes de uma release ser admitida.

O cache opcional CUDA Q8_1 permanece desligado nesta release. A carga
distribuída de qualificação encontrou uma liberação fora da ordem LIFO no pool
VMM desse cache; o próprio fork define `GGML_CUDA_Q8_1_CACHE=0` como retorno ao
comportamento anterior. O backend fixa esse valor tanto nos RPCs quanto no
coordenador, sem alterar pesos ou binários.

## Construção reproduzível

`make build` busca somente a revisão fixada e cria `build-sm75-64`. Depois da
construção, compare os três binários com `manifest.json`. Uma divergência cria
uma nova release qualificada; ela não deve ser corrigida alterando o hash no
host.

O modelo é identificado separadamente pelo SHA-256 do seu `SHA256SUMS`.
Assim, trocar pesos não se disfarça de recompilação do runtime, e recompilar o
runtime não se disfarça de troca de modelo.
