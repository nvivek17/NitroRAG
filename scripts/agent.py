import socket
import struct
import sys
import json
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

print("Loading Models...")
embedder=SentenceTransformer('all-MiniLM-L6-v2')
tokenizer=AutoTokenizer.from_pretrained("Xenova/llama2.c-strories15M")

VECTOR_DB_IP,VECTOR_DB_PORT='127.0.0.1',8080
INFERENCE_IP,INFERENCE_PORT='127.0.0.1',8081

def send_structured_query(sock,vector,subset_mask,r_min,r_max):
    meta=struct.pack('<Q f f',subset_mask,r_min,r_max)
    sock.sendall(meta)
    vec_data=struct.pack(f'<{len(vector)}f',*vector)
    sock.sendall(vec_data)

def recv_ints(sock):
    raw_len=sock.recv(4)
    if not raw_len:return []
    arr_len = struct.unpack('<i',raw_len)[0]
    raw_data=sock.recv(arr_len*4)
    return list(struct.unpack(f'<{arr_len}i',raw_data))

def extract_metadata(user_input):
    mask=0
    r_min,r_max=-1e9,1e9

    # Need to fill these according to data using regex and 
    years=re.findall(r'\b(20\d{2})\b',user_input)
    if(len(years) >=2):
        r_min,r_max=float(min(years)),float(max(years))
    elif len(years) ==1:
        r_min=r_max=float(years[0])
    
    #extract Categorical using fuzzy matching according to data but now as a example I will continue with direct matching
    if "security" in user_input.lower(): mask |= (1 << 0)
    if "bug" in user_input.lower(): mask |= (1 << 1)

    return mask, r_min, r_max

def ask_nitro(user_question):
    subset_mask, r_min, r_max = extract_metadata(user_question)    
    query_vector = embedder.encode(user_question).tolist() 
    
    db_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    db_sock.connect((VECTOR_DB_IP, VECTOR_DB_PORT))
    send_structured_query(db_sock, query_vector, subset_mask, r_min, r_max)
    context_tokens = recv_ints(db_sock)
    db_sock.close()

    question_tokens = tokenizer.encode(f" Context: ... Question: {user_question} Answer:", add_special_tokens=False)
    final_prompt = [1] + context_tokens + question_tokens # 1 is BOS

    inf_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    inf_sock.connect((INFERENCE_IP, INFERENCE_PORT))
    
    inf_sock.sendall(struct.pack('<i', len(final_prompt)))
    inf_sock.sendall(struct.pack(f'<{len(final_prompt)}i', *final_prompt))

    print("\n[NitroRAG Engine]: ", end="", flush=True)
    while True:
        raw_bytes = inf_sock.recv(4)
        if not raw_bytes: break 
        token_id = struct.unpack('<i', raw_bytes)[0]
        print(tokenizer.decode([token_id]), end="", flush=True)

    print("\n")
    inf_sock.close()

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question: ")
        if q == 'exit': break
        ask_nitro(q)

