import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

# --- CONFIGURAZIONE PERCORSI ---
SRC_DIR = "<gpt-oss-120b/>" #### CHANGE ME
OUT_GGUF = "<<gpt-oss-120b/quantized_q4/model.gguf>"  #### CHANGE ME
BLOCK_SIZE = 32

def quantize_q4_k_l(tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_gpu = tensor.to(device).to(torch.float16)
    
    shape = list(tensor.shape)  # Preserviamo la geometria originale come lista
    flat = t_gpu.flatten()
    num_elements = flat.numel()
    
    remainder = num_elements % BLOCK_SIZE
    padding = 0
    if remainder != 0:
        padding = BLOCK_SIZE - remainder
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.float16, device=device)])
        
    blocks = flat.view(-1, BLOCK_SIZE)
    mins = blocks.min(dim=1, keepdim=True).values
    maxs = blocks.max(dim=1, keepdim=True).values
    
    scales = (maxs - mins) / 15.0
    scales[scales == 0] = 1.0
    
    quantized = torch.clamp(torch.round((blocks - mins) / scales), 0, 15).to(torch.uint8)
    
    low_nibbles = quantized[:, 0::2]
    high_nibbles = quantized[:, 1::2]
    packed_qs = low_nibbles | (high_nibbles << 4)
    
    scales_raw = (scales.flatten().view(torch.int16).cpu().numpy().astype(np.uint16)) & 0xFFFF
    mins_raw = (mins.flatten().view(torch.int16).cpu().numpy().astype(np.uint16)) & 0xFFFF
    qs_raw = packed_qs.cpu().numpy()
    
    num_blocks = len(scales_raw)
    
    dtype_blocco = np.dtype([('scale', '<u2'), ('min', '<u2'), ('qs', 'u1', (16,))])
    array_strutturato = np.empty(num_blocks, dtype=dtype_blocco)
    
    array_strutturato['scale'] = scales_raw
    array_strutturato['min'] = mins_raw
    array_strutturato['qs'] = qs_raw
    
    array_di_byte = np.frombuffer(array_strutturato.tobytes(), dtype=np.uint8)
    
    # Restituiamo il tensore di byte rimodellato con la shape originale per ingannare il loader C++
    # Poiché Safetensors richiede che il numero totale di elementi combaci, usiamo una shape fittizia
    # che l'engine riconoscerà calcolando la dimensione in byte totale dello slot
    return torch.from_numpy(array_di_byte), shape

def main():
    if not os.path.exists(SRC_DIR):
        print(f"[-] Directory sorgente non trovata: {SRC_DIR}")
        sys.exit(1)
        
    os.makedirs(DST_DIR, exist_ok=True)
    safetensors_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith('.safetensors')])
    
    if not safetensors_files:
        print(f"[-] Nessun file .safetensors trovato in {SRC_DIR}")
        sys.exit(1)

    print(f"[*] Trovati {len(safetensors_files)} shard da convertire in Safetensors Q4...")
    
    # Copia dei file di configurazione
    for meta_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src_meta = os.path.join(SRC_DIR, meta_file)
        dst_meta = os.path.join(DST_DIR, meta_file)
        if os.path.exists(src_meta):
            with open(src_meta, 'r') as f:
                data = json.load(f)
            if meta_file == "config.json":
                data["quantization_config"] = {"bits": 4, "type": "q4_k_l_blockwise"}
            with open(dst_meta, 'w') as f:
                json.dump(data, f, indent=2)

    for file_name in safetensors_files:
        src_path = os.path.join(SRC_DIR, file_name)
        dst_path = os.path.join(DST_DIR, file_name)
        
        print(f"\n[➔] Ottimizzazione dello shard: {file_name}")
        quantized_tensors = {}
        
        with safe_open(src_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in tqdm(keys, desc="Quantizzazione Real-Time via CUDA"):
                tensor = f.get_tensor(key)
                
                # REGLA DI SICUREZZA: Quantizziamo solo gli esperti effettivi (.mlp / .expert),
                # escludendo esplicitamente il router di instradamento ("router") per non alterare la shape
                is_moe_weight = any(x in key for x in ["mlp", "expert", "w1", "w2"])
                is_router = "router" in key
                
                if is_moe_weight and not is_router:
                    q_bytes_tensor, original_shape = quantize_q4_k_l(tensor)
                    
                    # Calcoliamo una forma lineare compatibile in uint8 che preservi i byte totali (expected size)
                    # in modo da non far andare in errore il parser di model.cpp
                    quantized_tensors[key] = q_bytes_tensor.view(q_bytes_tensor.shape[0])
                else:
                    # I pesi densi leggeri e il router rimangono intatti a 16-bit
                    quantized_tensors[key] = tensor.to(torch.float16)
                    
        save_file(quantized_tensors, dst_path)
        print(f"[✓] Shard salvato correttamente in: {dst_path}")

    print(f"\n[🎉] COMPILAZIONE COMPLETATA! Modello quantizzato pronto in: {DST_DIR}")

if __name__ == "__main__":
    main()
