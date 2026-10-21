import os
import sys
import json
import struct
import torch
import numpy as np
from tqdm import tqdm
from safetensors import safe_open

# --- CONFIGURAZIONE PERCORSI ---
SRC_DIR = "<gpt-oss-120b/>" #### CHANGE ME
OUT_GGUF = "<<gpt-oss-120b/quantized_q4/model.gguf>"  #### CHANGE ME
BLOCK_SIZE = 32

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 2

def quantize_q4_k_l(tensor):
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)
    shape = tensor.shape
    flat = tensor.flatten()
    num_elements = flat.numel()
    
    remainder = num_elements % BLOCK_SIZE
    padding = 0
    if remainder != 0:
        padding = BLOCK_SIZE - remainder
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.float16, device=flat.device)])
        
    blocks = flat.view(-1, BLOCK_SIZE)
    mins = blocks.min(dim=1, keepdim=True).values
    maxs = blocks.max(dim=1, keepdim=True).values
    
    scales = (maxs - mins) / 15.0
    scales[scales == 0] = 1.0
    
    quantized = torch.clamp(torch.round((blocks - mins) / scales), 0, 15).to(torch.uint8)
    
    # Compattiamo 2 elementi a 4 bit in 1 singolo byte
    low_nibbles = quantized[:, 0::2]
    high_nibbles = quantized[:, 1::2]
    packed_qs = low_nibbles | (high_nibbles << 4)
    
    return packed_qs, scales.flatten(), mins.flatten(), shape

def write_gguf_string(f, s):
    encoded = s.encode('utf-8')
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)

def main():
    if not os.path.exists(SRC_DIR):
        print(f"[-] Directory sorgente non trovata: {SRC_DIR}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(OUT_GGUF), exist_ok=True)
    safetensors_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith('.safetensors')])
    
    if not safetensors_files:
        print(f"[-] Nessun file .safetensors trovato in {SRC_DIR}")
        sys.exit(1)

    print(f"[*] Analisi di {len(safetensors_files)} shard safetensors per unificazione GGUF...")
    
    # Caricamento configurazione per i metadati dell'header
    config_path = os.path.join(SRC_DIR, "config.json")
    layers, n_experts, hidden_dim = 36, 128, 2880
    if os.path.exists(config_path):
        with open(config_path, 'r') as cf:
            cfg = json.load(cf)
            layers = cfg.get("num_hidden_layers", layers)
            n_experts = cfg.get("num_local_experts", n_experts)
            hidden_dim = cfg.get("hidden_size", hidden_dim)

    # Raccolta e blindatura della struttura dei tensori
    tensor_meta = []
    print("[*] Mappatura dell'indice dei tensori...")
    for f_name in safetensors_files:
        p = os.path.join(SRC_DIR, f_name)
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor_meta.append((k, p))

    if not tensor_meta:
        print("[-] Errore critico: Nessun tensore trovato mappando i file safetensors.")
        sys.exit(1)

    print(f"[+] Indicizzati {len(tensor_meta)} tensori pronti alla conversione.")

    # Apertura del file GGUF finale per la scrittura binaria
    with open(OUT_GGUF, "wb") as f_out:
        # 1. Scrittura GGUF Header
        f_out.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensor_meta), 4))
        
        # 2. Scrittura KV Metadata obbligatori per l'engine
        write_gguf_string(f_out, "general.architecture")
        f_out.write(struct.pack("<I", 8)) # String type
        write_gguf_string(f_out, "gpt-oss")
        
        write_gguf_string(f_out, "gptoss.layer_count")
        f_out.write(struct.pack("<I", 4)) # Int32 type
        f_out.write(struct.pack("<i", layers))
        
        write_gguf_string(f_out, "gptoss.expert_count")
        f_out.write(struct.pack("<I", 4))
        f_out.write(struct.pack("<i", n_experts))

        write_gguf_string(f_out, "general.alignment")
        f_out.write(struct.pack("<I", 4))
        f_out.write(struct.pack("<i", 32))

        # 3. Scrittura dell'indice dei Tensori (Tensor Info)
        current_offset = 0
        tensor_data_entries = []
        
        print("[*] Generazione del dizionario geometrico GGUF...")
        for k, p in tensor_meta:
            write_gguf_string(f_out, k)
            
            with safe_open(p, framework="pt", device="cpu") as f_tensor:
                t = f_tensor.get_tensor(k)
                shape = list(t.shape)
                
            f_out.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f_out.write(struct.pack("<Q", dim))
                
            # Identifichiamo il tipo di quantizzazione binaria
            is_moe = any(x in k for x in ["mlp", "expert", "w1", "w2"])
            t_type = 2 if is_moe else 1 
            f_out.write(struct.pack("<I", t_type))
            
            # Assegniamo l'offset provvisorio relativo al corpo dei dati binari
            f_out.write(struct.pack("<Q", current_offset))
            
            # Calcolo preventivo della dimensione binaria post-quantizzazione
            num_elements = 1
            for d in shape: num_elements *= d
            if is_moe:
                num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                t_size = num_blocks * 20
            else:
                t_size = num_elements * 2 # FP16 = 2 byte
                
            tensor_data_entries.append((k, p, is_moe, t_size))
            current_offset += t_size
            current_offset = ((current_offset + 31) // 32) * 32

        # Allineamento finale dell'header prima della scrittura del blocco dati
        pad_header = (f_out.tell() + 31) // 32 * 32
        f_out.write(b'\x00' * (pad_header - f_out.tell()))

        # 4. Scrittura fisica dei dati dei Tensori (Quantizzazione Real-Time Vettorizzata)
        for k, p, is_moe, expected_size in tqdm(tensor_data_entries, desc="Compressione e Scrittura Tensori GGUF"):
            start_pos = f_out.tell()
            with safe_open(p, framework="pt", device="cpu") as f_tensor:
                t = f_tensor.get_tensor(k)
                
            if is_moe:
                packed_qs, scales, mins, _ = quantize_q4_k_l(t)
                
                # VETTORIZZAZIONE: Trasformiamo i tensori in visualizzazioni int16 immediate con maschera
                # === VERSIONE CORRETTA E CONVERTITA IN UINT16 ===
                scales_raw = (scales.half().view(torch.int16).numpy().astype(np.uint16)) & 0xFFFF
                mins_raw = (mins.half().view(torch.int16).numpy().astype(np.uint16)) & 0xFFFF
                qs_raw = packed_qs.numpy()
                
                num_blocks = len(scales)
                
                # Sfruttiamo un array strutturato NumPy per allineare la memoria a livello C
                dtype_blocco = np.dtype([('scale', '<u2'), ('min', '<u2'), ('qs', 'u1', (16,))])
                array_strutturato = np.empty(num_blocks, dtype=dtype_blocco)
                
                array_strutturato['scale'] = scales_raw.astype('<u2')
                array_strutturato['min'] = mins_raw.astype('<u2')
                array_strutturato['qs'] = qs_raw
                
                f_out.write(array_strutturato.tobytes())
            else:
                f_out.write(t.to(torch.float16).numpy().tobytes())
                
            # Forziamo il padding a 32 byte tra un tensore e l'altro
            end_pos = f_out.tell()
            pad_needed = ((end_pos + 31) // 32 * 32) - end_pos
            f_out.write(b'\x00' * pad_needed)

    print(f"\n[🎉] COMPILAZIONE COMPLETATA! File GGUF monolitico Q4_K_L salvato in: {OUT_GGUF}")
    print(f"[i] Dimensione finale stimata su disco: circa 35-40 GB.")

if __name__ == "__main__":
    main()
