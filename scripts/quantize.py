import struct
import numpy as np
import os

Q_BLOCK_SIZE = 32

def quantize_q8(weights):
    """Takes a 1D numpy array of floats, chunks it by 32, and returns Q8 bytes."""
    out_bytes = bytearray()
    for i in range(0, len(weights), Q_BLOCK_SIZE):
        block = weights[i : i + Q_BLOCK_SIZE]
        abs_max = np.max(np.abs(block))
        scale = abs_max / 127.0 if abs_max != 0 else 1e-9
        
        # Write the scale (4 bytes float)
        out_bytes += struct.pack('f', scale)
        
        # Quantize and write the 32 weights (32 bytes int8)
        for w in block:
            q_w = round(w / scale)
            q_w = max(-128, min(127, int(q_w)))
            out_bytes += struct.pack('b', q_w)
    return out_bytes

print("Starting Quantization Process...")

with open('stories110M.bin', 'rb') as f_in, open('stories110M_q8.bin', 'wb') as f_out:

    header = f_in.read(28)
    f_out.write(header)
    
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', header)
    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size

    def copy_fp32(count):
        f_out.write(f_in.read(count * 4))

    def convert_to_q8(count):
        floats = np.frombuffer(f_in.read(count * 4), dtype=np.float32)
        f_out.write(quantize_q8(floats))

    print("Quantizing tensors...")
    copy_fp32(vocab_size * dim)
    copy_fp32(n_layers * dim)
    
    convert_to_q8(n_layers * dim * dim)     # wq
    convert_to_q8(n_layers * dim * kv_dim)  # wk
    convert_to_q8(n_layers * dim * kv_dim)  # wv
    convert_to_q8(n_layers * dim * dim)     # wo

    copy_fp32(n_layers * dim)

    convert_to_q8(n_layers * dim * hidden_dim) # w1
    convert_to_q8(n_layers * hidden_dim * dim) # w2
    convert_to_q8(n_layers * dim * hidden_dim) # w3
    
    copy_fp32(dim)
    copy_fp32(seq_len * head_size // 2) # freq_cis_real
    copy_fp32(seq_len * head_size // 2) # freq_cis_imag
    
    print("Quantization Complete! Created stories110M_q8.bin")