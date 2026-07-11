#pragma once
#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <netinet/in.h>
#include "pagedMemory.hpp"
#include "NitroRPC.hpp"
#include "model_q8.cpp"

using namespace std;

mutex queue_mutex;
queue<Request*> incoming_requests;

void tcp_listener_thread(){
    int server_fd=socket(AF_INET,SOCK_STREAM,0);
    int opt=1;setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in address;
    address.sin_family=AF_INET;
    address.sin_addr.s_addr=INADDR_ANY;
    address.sin_port=htons(8081);
    bind(server_fd,(struct sockaddr*)&address,sizeof(address));
    listen(server_fd,100);
    cout<<"[Inference Server] Listening on Port 8081..."<<endl;
    int req_counter=0;
    while(true){
        int client_sock=accept(server_fd,nullptr,nullptr);
        vector<int> prompt_tokens;
        if(NitroRPC::recv_ints(client_sock,prompt_tokens)){
            Request* req=new Request(req_counter++,prompt_tokens);
            req->client_socket=client_sock;
            lock_guard<mutex> lock(queue_mutex);
            incoming_requests.push(req);
        }
    }
}

int main(){
     cout << "Loading Int8 Quantized Model (stories110M_q8.bin)..." << endl;
    int fd = open("stories110M_q8.bin", O_RDONLY);
    if(fd < 0) { cerr << "Open Failed " << endl; exit(1); }
    struct stat st; fstat(fd, &st);
    void* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    int* header = (int*)data;
    Config_Transformer c;
    c.dim = header[0]; c.hidden_dim = header[1]; c.n_layers = header[2];
    c.n_heads = header[3]; c.n_kv_heads = header[4]; c.vocab_size = header[5]; c.seq_len = header[6];
    int head_size = c.dim / c.n_heads;
    int kv_dim = c.n_kv_heads * head_size;
    uint8_t* byte_ptr = (uint8_t*)data + 28;
    Transformer_Weights w;
    w.token_embedding_table = (float*)byte_ptr; byte_ptr += c.vocab_size * c.dim * sizeof(float);
    w.rms_attn_weights = (float*)byte_ptr;      byte_ptr += c.n_layers * c.dim * sizeof(float);
    int Q_BLOCK_SIZE = 32;
    w.wq = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.dim * c.dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);
    w.wk = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.dim * kv_dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);
    w.wv = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.dim * kv_dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);
    w.wo = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.dim * c.dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);

    w.rms_ffn_weight = (float*)byte_ptr; byte_ptr += c.n_layers * c.dim * sizeof(float);
    w.w1 = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.hidden_dim * c.dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);
    w.w2 = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.dim * c.hidden_dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);
    w.w3 = (BlockQ8*)byte_ptr; byte_ptr += (c.n_layers * c.hidden_dim * c.dim) / Q_BLOCK_SIZE * sizeof(BlockQ8);

    w.rms_final_weight = (float*)byte_ptr; byte_ptr += c.dim * sizeof(float);
    w.freq_cis_real = (float*)byte_ptr;    byte_ptr += (c.seq_len * head_size / 2) * sizeof(float);
    w.freq_cis_imag = (float*)byte_ptr;    byte_ptr += (c.seq_len * head_size / 2) * sizeof(float);
    w.wcls = (float*)w.token_embedding_table; 

    cout << " Mapped " << st.st_size / (1024*1024) << " MB. Hardware Aligned" << endl;
    int MAX_BATCH = 512; 
    RunState s;
    s.x = (float*)calloc(MAX_BATCH * c.dim, sizeof(float));
    s.xb = (float*)calloc(MAX_BATCH * c.dim, sizeof(float));
    s.xb2 = (float*)calloc(MAX_BATCH * c.dim, sizeof(float));
    s.hb = (float*)calloc(MAX_BATCH * c.hidden_dim, sizeof(float));
    s.hb2 = (float*)calloc(MAX_BATCH * c.hidden_dim, sizeof(float));
    s.q = (float*)calloc(MAX_BATCH * c.dim, sizeof(float));
    s.k = (float*)calloc(MAX_BATCH * kv_dim, sizeof(float));
    s.v = (float*)calloc(MAX_BATCH * kv_dim, sizeof(float));
    s.logits = (float*)calloc(MAX_BATCH * c.vocab_size, sizeof(float));
    s.attn = (float*)calloc(MAX_BATCH * c.n_heads * c.seq_len, sizeof(float));

    int TOTAL_BLOCKS = 1000; 
    KVBlock_Manager mem_mgr(TOTAL_BLOCKS, c.n_layers, kv_dim);
    PrefixCacheManager cache_mgr(mem_mgr, TOTAL_BLOCKS);

    Continuous_Scheduler scheduler;
    thread listener(tcp_listener_thread);
    cout << "[Engine] Continuous Batching Loop Started..." << endl;

    while (true) {
        queue_mutex.lock();
        while (!incoming_requests.empty()) {
            scheduler.add_request(incoming_requests.front());
            incoming_requests.pop();
        }
        queue_mutex.unlock();
        if (scheduler.has_active_requests()) {
            scheduler.step(&c, &w, &s, mem_mgr, cache_mgr); 
            for (Request* req : scheduler.running_batch) {
                if (req->just_generated_new_token) {
                    int tok = req->generated_tokens.back();
                    send(req->client_socket, &tok, sizeof(int), 0);
                    req->just_generated_new_token = false;
                }
                if (req->is_finished) {
                    close(req->client_socket); 
                }
            }
        } else {
            usleep(1000);
        }
    }
    return 0;
}