# gptoss-ext

Motore di inferenza CPU autonomo per **gpt-oss-120b** (GQA + sink + strati a finestra alternati, MoE 128x4 in MXFP4). Note e incognite: [docs/gptoss_porting.md](docs/gptoss_porting.md).

    sh build.sh                                   # compila e lancia i test sul checkpoint minuscolo (rigenerabile: python3 tools/tiny_gptoss.py tests/tiny)
    ./build-out/gptoss_run --model SNAPSHOT --ctx 8192 --selfcheck
    ./build-out/gptoss_run --model SNAPSHOT --prompt-ids 200006,17360,200008 --n-predict 32
    ./build-out/gptoss_run --model SNAPSHOT --serve --ctx 32768 --prefill-chunk 512 --kv-unified \
        --kv-checkpoint /var/tmp/gptoss.kv --kv-rollback 8192 --eos-ids 200002,200012
    ./build-out/gptoss_run --model SNAPSHOT --selfcheck --vram-pct 90 --vram-total-mib 24576   # piano, non esecuzione

    # dentro il checkout di Strata (cartella accanto a dsv4_ext/): UI + API di Strata sopra gptoss_run
    python3 tools/serve_gptoss_strata.py --model SNAPSHOT --port 8095 --ctx 32768 --kv-checkpoint /var/tmp/gptoss.kv

    # CUDA (binario separato, mai nello stesso del backend emulato):
    sh build.sh --cuda                                           # CUDA_ARCH=sm_89 (o native), NVCC=/percorso/nvcc
    ./build-out/gptoss_run_cuda --model SNAPSHOT --cuda --vram-pct 90 --serve --ctx 32768 --kv-checkpoint /var/tmp/gptoss.kv
    #   --vram-pct: % della scheda (cudaMemGetInfo); --vram-total-mib N per forzare la dimensione; --gpu-slots N per limitare gli slot

SNAPSHOT = cartella con `config.json` e `model-0000x-of-00014.safetensors` (la sottocartella `original/` non serve).
Il motore lavora su id di token (Harmony e tokenizer o200k_harmony stanno fuori).

| Area | Stato | Evidenza |
| --- | --- | --- |
| MXFP4, BF16, lettore multi-shard, config | OK su sintetico | test |
| Forward (GQA, sink, finestra, YaRN, router, swiglu, bias) vs numpy float64 | OK su sintetico | 4 varianti, \|diff\| max 4,3e-5, argmax identici; le varianti differiscono davvero |
| Prefill a batch (chunk 1/3/64, anello pieno e anello che si riavvolge) | OK su sintetico | \|diff\| 6e-6 |
| Riuso prefisso, checkpoint KV (anche con anello), rifiuto se cambia configurazione, rifiuto del riavvolgimento oltre l'anello | OK su sintetico | test + prova `--serve` (`40 of 44 cached`, riavvio `30 of 32`) |
| Modello vero (qualsiasi cosa) | **NON RUN** | 60,8 GiB, nessuna rete verso HF nel sandbox |
| Incognite 1-4, 6-8 della tabella "Non noto" | **APERTE** | servono i logit di riferimento |
| 32k token/s di prefill | **NON raggiunto** | ~0,33 PFLOP/s necessari |
| CUDA (`src/gpu_cuda.cu`, pool di expert in VRAM con LRU, matmat BF16/MXFP4, prefill a batch su device) | **codice scritto, MAI compilato ne' eseguito** | il percorso e' provato solo sul device emulato (test `device(emulated)`, scritti ma non lanciati dopo l'ultima modifica) |
| GPU, memoria a livelli, GEMM di prefill | NON FATTI | |
