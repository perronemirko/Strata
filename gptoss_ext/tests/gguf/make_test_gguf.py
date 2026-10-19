#!/usr/bin/env python3
# Crea un GGUF di prova (v3) con tensori F32, Q8_0 e Q4_K, più i byte attesi calcolati qui in Python,
# indipendentemente dal C++. Uso: python3 make_test_gguf.py [cartella_output]
import os, random, struct, sys

out = sys.argv[1] if len(sys.argv) > 1 else "."
ALIGN = 64
rng = random.Random(7)

def s(x):
    b = x.encode()
    return struct.pack('<Q', len(b)) + b

def kv_u32(k, v): return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)
def kv_str(k, v): return s(k) + struct.pack('<I', 8) + s(v)
def kv_f32(k, v): return s(k) + struct.pack('<I', 6) + struct.pack('<f', v)
def kv_arr_str(k, vals):
    return s(k) + struct.pack('<I', 9) + struct.pack('<I', 8) + struct.pack('<Q', len(vals)) + b''.join(s(v) for v in vals)

kvs = [kv_str('general.architecture', 'test'), kv_u32('general.alignment', ALIGN),
       kv_f32('test.pi', 3.14), kv_arr_str('test.words', ['uno', 'due', 'tre'])]

# F32: 16 valori 0..15, dims [8,2]
f32_blob = struct.pack('<16f', *[float(i) for i in range(16)])

# Q8_0: 4 blocchi da 32 elementi (128), dims [64,2]. Blocco = d fp16 + 32 int8. Valore = q * d.
q8_blob = b''
q8_expected = b''
for _ in range(4):
    d = struct.unpack('<e', struct.pack('<e', rng.uniform(0.01, 2.0)))[0]
    qs = [rng.randint(-128, 127) for _ in range(32)]
    q8_blob += struct.pack('<e', d) + struct.pack('<32b', *qs)
    q8_expected += b''.join(struct.pack('<f', float(q) * d) for q in qs)

# Q4_K: 2 super-blocchi da 256 (512), dims [256,2]. Byte casuali: il test confronta i byte grezzi.
q4k_blob = bytes(rng.randrange(256) for _ in range(288))

tensors = [('a.f32', [8, 2], 0, f32_blob), ('b.q8', [64, 2], 8, q8_blob), ('c.q4k', [256, 2], 12, q4k_blob)]

infos = b''
data = b''
off = 0
for name, dims, typ, blob in tensors:
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    infos += s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims)
    infos += struct.pack('<I', typ) + struct.pack('<Q', off)
    data = data.ljust(off, b'\0') + blob
    off += len(blob)

header = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(tensors)) + struct.pack('<Q', len(kvs))
header += b''.join(kvs) + infos
header = header.ljust((len(header) + ALIGN - 1) // ALIGN * ALIGN, b'\0')

os.makedirs(out, exist_ok=True)
for name, content in [('test.gguf', header + data), ('expected_f32.bin', f32_blob),
                      ('expected_q8.bin', q8_expected), ('expected_q4k.bin', q4k_blob)]:
    open(os.path.join(out, name), 'wb').write(content)
print('ok: test.gguf, expected_f32.bin, expected_q8.bin, expected_q4k.bin in ' + out)
