#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <list>
using namespace std;

const int BLOCK_SIZE=16;

const uint64_t FNV_OFFSET_BASIS=14695981039346656037ULL;
const uint64_t FNV_PRIME=1099511628211ULL;

inline uint64_t hash_block_state(uint64_t parent_hash,const vector<int>& tokens){
    uint64_t hash=parent_hash;
    for(int token : tokens){
        const uint8_t* ptr =reinterpret_cast<const uint8_t*>(&token);
        for(int i=0;i<sizeof(int);i++){
            hash^=ptr[i];
            hash*=FNV_PRIME;
        }
    }
    return hash;
}



struct KVBlock{
    int physical_id;
    int num_tokens;
    int ref_count;

    int parent_id;
    int child_count;
    float* k_ptr;
    float* v_ptr;
};

class KVBlock_Manager;
class PrefixCacheManager;

class KVBlock_Manager{
    private:
    float* global_k_pool;
    float* global_v_pool;
    int kv_dim;
    int n_layers;
    
    vector<KVBlock> blocks;
    vector<int> free_list;
    
    public:
    KVBlock_Manager(int total_blocks,int layers,int _kv_dim);
    int allocate_block(PrefixCacheManager& cache_mgr);
    void inc_ref_count(int id,PrefixCacheManager& cache_mgr);
    int get_free_count();
    void free_block(int id,PrefixCacheManager& cache_mgr);
    KVBlock& get_block(int id);
    void force_free_block(int id);
    ~KVBlock_Manager();
};


class PrefixCacheManager{
    private:
    unordered_map<uint64_t,int> cache_map;
    unordered_map<int,uint64_t> reverse_map;
    list<int> evictable_leaves;
    unordered_map<int,list<int>::iterator> leaf_pointers;
    
    int total_system_blocks;
    int min_free_blocks;
    int taget_free_blocks;
    public:
    KVBlock_Manager& mem_mgr;
    
    PrefixCacheManager(KVBlock_Manager& mm,int total_blocks);
    void evaluate_evictability(int phy_id);
    void insert_cache(uint64_t block_hash, int physical_block_id,int parent_phy_id);
    void evict_lru_block();
    bool is_cached(int phy_id);
    int check_cache(uint64_t block_hash);
};


KVBlock_Manager::KVBlock_Manager(int total_blocks,int layers,int _kv_dim){
    n_layers=layers;
    kv_dim=_kv_dim;

    size_t block_floats=(size_t)n_layers*BLOCK_SIZE*kv_dim;
    size_t total_floats=total_blocks*block_floats;

    global_k_pool=(float*) aligned_alloc(64,total_floats*sizeof(float));
    global_v_pool=(float*) aligned_alloc(64,total_floats*sizeof(float));

    if(!global_k_pool||!global_v_pool){cerr<<"Error Failed to allocate KV Cache Pool"<<endl;exit(1);}
    blocks.resize(total_blocks);

    for(int i=0;i<total_blocks;i++){
        blocks[i].physical_id=i;
        blocks[i].num_tokens=0;
        blocks[i].ref_count=0;
        blocks[i].parent_id=-1;
        blocks[i].child_count=0;
        blocks[i].k_ptr=global_k_pool+(i*block_floats);
        blocks[i].v_ptr=global_v_pool+(i* block_floats);

        free_list.push_back(i);
    }
    //cout<< "Initialized PagedAttention Pool "<<total_blocks <<" blocks"<<endl;
}

int KVBlock_Manager::allocate_block(PrefixCacheManager& cache_mgr){
    if(free_list.empty()){
        cache_mgr.evict_lru_block();
        if(free_list.empty()){
            cerr<<"Out of Memory. Eviction Failed "<<endl;exit(1);
        }
    }
    int id=free_list.back();
    free_list.pop_back();

    blocks[id].ref_count=1;
    blocks[id].num_tokens=0;
    blocks[id].parent_id=-1;
    blocks[id].child_count=0;
    return id;
}
void KVBlock_Manager::inc_ref_count(int id,PrefixCacheManager& cache_mgr){
    blocks[id].ref_count++;
    cache_mgr.evaluate_evictability(id);
}

int KVBlock_Manager::get_free_count(){
    return free_list.size();
}
void KVBlock_Manager::free_block(int id,PrefixCacheManager& cache_mgr){
    blocks[id].ref_count--;
    cache_mgr.evaluate_evictability(id);
    if(blocks[id].ref_count==0&& !cache_mgr.is_cached(id)){
        blocks[id].num_tokens=0;
        free_list.push_back(id);
    }
}
void KVBlock_Manager::force_free_block(int id){
    blocks[id].num_tokens=0;
    blocks[id].ref_count=0;
    blocks[id].parent_id=-1;
    blocks[id].child_count=0;
    free_list.push_back(id);
}


KVBlock& KVBlock_Manager::get_block(int id){
    return blocks[id];
}
KVBlock_Manager::~KVBlock_Manager(){
    free(global_k_pool);
    free(global_v_pool);
}


PrefixCacheManager::PrefixCacheManager(KVBlock_Manager& mm,int total_blocks):mem_mgr(mm){
    total_system_blocks=total_blocks;
    min_free_blocks=std::max(1,(int)(total_blocks*0.05));
    taget_free_blocks=std::max(2,(int)(total_blocks*0.2));
}
bool PrefixCacheManager:: is_cached(int phy_id){
    return (reverse_map.count(phy_id)!=0);
}
    void PrefixCacheManager::evaluate_evictability(int phy_id){
        if(!is_cached(phy_id)) return;
        KVBlock& b=mem_mgr.get_block(phy_id);
        
        bool is_evictable=(b.ref_count==0 && b.child_count == 0);
        bool in_lru=(leaf_pointers.count(phy_id)!=0);
        
        if(is_evictable&& !in_lru){
            evictable_leaves.push_front(phy_id);
            leaf_pointers[phy_id]=evictable_leaves.begin();
        }
        else if(!is_evictable&& in_lru){
            evictable_leaves.erase(leaf_pointers[phy_id]);
            leaf_pointers.erase(phy_id);
        }
    }
    int PrefixCacheManager::check_cache(uint64_t block_hash){
        if(cache_map.count(block_hash)!=0){
            return cache_map[block_hash];
        }
        return -1;
    }
    void PrefixCacheManager::insert_cache(uint64_t block_hash, int physical_block_id,int parent_phy_id){
        cache_map[block_hash]=physical_block_id;
        reverse_map[physical_block_id]=block_hash;
        
        KVBlock& b=mem_mgr.get_block(physical_block_id);
        b.parent_id=parent_phy_id;
        b.child_count=0;
        if(parent_phy_id!=-1){
            mem_mgr.get_block(parent_phy_id).child_count++;
            evaluate_evictability(parent_phy_id);
        }
        evaluate_evictability(physical_block_id);
    }
    void PrefixCacheManager::evict_lru_block(){
        if(mem_mgr.get_free_count()>=min_free_blocks) return;
        //cout<<"Memory below 5% , Starting Bulk Topological Eviction..."<<endl;
        while(mem_mgr.get_free_count() < taget_free_blocks){
            if(evictable_leaves.empty()){
                if(mem_mgr.get_free_count()>0) break;
                cerr<<"High Traffic.No evictable leaves left!"<<endl;exit(1);
            }
            int evict_id=evictable_leaves.back();
            evictable_leaves.pop_back();
            leaf_pointers.erase(evict_id);
            
            uint64_t hash = reverse_map[evict_id];
            cache_map.erase(hash);
            reverse_map.erase(evict_id);
            int parent_id=mem_mgr.get_block(evict_id).parent_id;
            if(parent_id !=-1){
                mem_mgr.get_block(parent_id).child_count--;
                evaluate_evictability(parent_id);
            }
            mem_mgr.force_free_block(evict_id);
        }
        cerr<<"Eviction Complete. Memory restored to 20%"<<endl;
    }



struct Sequence{
    int seq_id;
    vector<int> block_table;
    int logical_token_count;
    uint64_t current_hash;
    vector<int> token_buffer;

    Sequence(int id){
        seq_id=id;
        logical_token_count=0;
        current_hash=FNV_OFFSET_BASIS;
    }
    void allocate_next_token(int token,KVBlock_Manager& manager,PrefixCacheManager& cache_mgr){
       
       if(token_buffer.size()==BLOCK_SIZE){
        uint64_t new_hash=hash_block_state(current_hash,token_buffer);
        current_hash=new_hash;
        int finished_physical_id=block_table.back();
        int parent_id=(block_table.size()>1)? block_table[block_table.size()-2]:-1;
        cache_mgr.insert_cache(current_hash,finished_physical_id,parent_id);
        token_buffer.clear();
       }
       
        if(logical_token_count% BLOCK_SIZE ==0){
            int new_block_id=manager.allocate_block(cache_mgr);
            block_table.push_back(new_block_id);
        }
        token_buffer.push_back(token);
        int curr_physical_block=block_table.back();
        manager.get_block(curr_physical_block).num_tokens++;
        logical_token_count++; 
    }
    void flush_cache(PrefixCacheManager& cache_mgr){
        if(token_buffer.size()==BLOCK_SIZE){
            uint64_t new_hash=hash_block_state(current_hash,token_buffer);
            current_hash=new_hash;
             int parent_id=(block_table.size()>1)? block_table[block_table.size()-2]:-1;
            int finished_phy_id=block_table.back();
            cache_mgr.insert_cache(current_hash,finished_phy_id,parent_id);
        }
    }
    void release(KVBlock_Manager& manager,PrefixCacheManager& cache_mgr){
        for(int block_id : block_table){
            manager.free_block(block_id,cache_mgr);
        }
        block_table.clear();
        token_buffer.clear();
        logical_token_count=0;
        current_hash=FNV_OFFSET_BASIS;
        // //cout<<"Sequence"<< seq_id << "Has been completed. Memory Released"<<endl;
    }

};

struct Request{
    int req_id;
    vector<int> prompt_tokens;
    vector<int> generated_tokens;
    int client_socket=-1;
    bool just_generated_new_token=false;
    int prefill_idx=0;
    bool is_prefill_done=false;
    bool is_finished =false;
    int next_token_to_process=-1;
    Sequence seq;

    Request(int id, const vector<int>& prompt): req_id(id),prompt_tokens(prompt),seq(id){}
};


class Continuous_Scheduler{
 private:
 queue<Request*> waiting_queue;
 vector<Request*> running_batch;
 int max_tokens_per_batch=256;

public:
    void add_request(Request* req){
        req->next_token_to_process=req->prompt_tokens[0];
        waiting_queue.push(req);
    }
    bool has_active_requests(){
        return !waiting_queue.empty() || !running_batch.empty();
    }
    void step(Config_Transformer*c, Transformer_Weights* w,RunState*s,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr){
        for( auto it=running_batch.begin();it!= running_batch.end();){
            if((*it)->is_finished){
                //cout<<"[Continuous Scheduler] Request "<< (*it)->req_id<<" finished. Memory released "<<endl;
                (*it)->seq.release(mem_mgr,cache_mgr);
                it=running_batch.erase(it);
            }
            else{
                it++;
            }
        }
        int token_budget=max_tokens_per_batch;

        vector<int> decode_tokens;
        vector<int> decode_positions;
        vector<Request*> decode_reqs;

        for(Request* req : running_batch){
            if(req->is_prefill_done){
                decode_tokens.push_back(req->next_token_to_process);
                decode_positions.push_back(req->seq.logical_token_count);
                decode_reqs.push_back(req);
                token_budget-=1;
            }
        }
        Request* active_prefill_req=nullptr;
        vector<int> prefill_chunk;
        while(!waiting_queue.empty() && token_budget >0){
            Request* req =waiting_queue.front();
            while(req->prefill_idx+BLOCK_SIZE <= req->prompt_tokens.size() && req->seq.token_buffer.empty()){
                vector<int> p_chunk(req->prompt_tokens.begin()+req->prefill_idx,req->prompt_tokens.begin()+req->prefill_idx+BLOCK_SIZE);
                uint64_t hash= hash_block_state(req->seq.current_hash,p_chunk);
                int hit_id=cache_mgr.check_cache(hash);

                if(hit_id !=-1){
                    mem_mgr.inc_ref_count(hit_id,cache_mgr);
                    req->seq.block_table.push_back(hit_id);
                    req->seq.logical_token_count+=BLOCK_SIZE;
                    req->seq.current_hash=hash;
                    req->prefill_idx+=BLOCK_SIZE;
                    // //cout<<"[Scheduler] Request "<< req->req_id<< " prefix cache HIT "<< endl;
                }else{break;}
            }
            int remaining_prompt=req->prompt_tokens.size()-req->prefill_idx;
            int chunk_size=min(remaining_prompt,token_budget);
            if(chunk_size>0){
                prefill_chunk.assign(req->prompt_tokens.begin()+ req->prefill_idx,req->prompt_tokens.begin()+ req->prefill_idx+chunk_size);
                active_prefill_req=req;
            }
            if(req->prefill_idx+chunk_size == req->prompt_tokens.size()){
                req->is_prefill_done=true;
                running_batch.push_back(req);
                waiting_queue.pop();
            }
            else{
                req->prefill_idx+=chunk_size;
            }
            break;
        }

        if(active_prefill_req != nullptr && !prefill_chunk.empty()){
            //cout<< "Running Prefill Chunk ("<<prefill_chunk.size()<< " tokens) for Req"<< active_prefill_req->req_id<<endl;
            float* prefill_logits=forward_prefill_chunk(prefill_chunk,active_prefill_req->seq.logical_token_count,c,w,s,active_prefill_req->seq,mem_mgr,cache_mgr);
            if(active_prefill_req->is_prefill_done){
                int next_token=argmax(prefill_logits,c->vocab_size);
                active_prefill_req->next_token_to_process=next_token;
                active_prefill_req->generated_tokens.push_back(next_token);
            }
        }
        if(!decode_tokens.empty()){
            //cout<< " Running Decode Batch ("<< decode_tokens.size()<< " users)"<<endl;
            float* batch_logits=forward_decode_batch(decode_tokens,decode_positions,decode_reqs,c,w,s,mem_mgr,cache_mgr);
            for(int b=0;b<decode_reqs.size();b++){
                int next_token=argmax(batch_logits+(b* c->vocab_size),c->vocab_size);
                decode_reqs[b]->next_token_to_process=next_token;
                decode_reqs[b]->generated_tokens.push_back(next_token);
                if(decode_reqs[b]->generated_tokens.size() >=20){
                    decode_reqs[b]->is_finished=true;
                }
            }
        }
    }
};