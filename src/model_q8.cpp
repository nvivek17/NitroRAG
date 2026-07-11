#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/mman.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;
struct Config_Transformer{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};
const int Q_BLOCK_SIZE=32;
struct BlockQ8{
    float scale;
    int8_t weights[Q_BLOCK_SIZE];
};
struct Transformer_Weights{
    float* token_embedding_table; //(vocab_size,dim)
    
    float* rms_attn_weights;//(layer,dim)
    BlockQ8* wq;//(layer,dim,dim)
    BlockQ8* wk;//(layer,dim,dim)
    BlockQ8* wv;//(layer,dim,dim)
    BlockQ8* wo;//(layer,dim,dim)
    
    float* rms_ffn_weight;// (layer,dim)
    BlockQ8* w1;// (layer, hidden_dim,dim)
    BlockQ8* w2;// (layer, dim, hidden_dim)
    BlockQ8* w3;// (layer, hidden_dim,dim)
    
    float* freq_cis_real;
    float* freq_cis_imag;
    float* rms_final_weight; //(dim,1)
    float* wcls; //(vocab_size,dim) -classifier weights
};


struct RunState{
    float* x; //current activation (dim,) 
    float* xb;// activation inside residual branch (dim,)
    float* xb2;// addition buffer (dim,)
    float* hb; //buffer for hidden dim in FFN (hidden_dim,)
    float* hb2;//buffer for hidden dimension in FNN (hidden_dim,)
    
    float* q;//(dim,)
    float* k;//(dim,)
    float* v;//(dim,)
    float* attn;// buffer for scores (n_heads,seq_len)
    
    float* logits;// (vocab_size,)
    //KV Cache
    float* key_cache; //(layer,seq_len,dim)
    float* value_cache;//(layer.seq_len,dim)
};
class KVBlock_Manager;
class PrefixCacheManager;
struct Sequence;
struct Request;
float* forward_prefill_chunk(const vector<int>& tokens, int start_pos,Config_Transformer*c,Transformer_Weights* w,RunState*s,Sequence& seq,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr);
int argmax(float* prob, int size);
float* forward_decode_batch(const vector<int>& batch_tokens, const vector<int>& batch_positions,vector<Request*>& batch_reqs,Config_Transformer*c,Transformer_Weights* w,RunState*s,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr);

#include "pagedMemory.hpp"
void matmul(float* out,float* x,float*w,int n,int d){
    
    // w (d x n) @ x(n x 1) -> out(d x 1)
    #pragma omp parallel for
    for(int i=0;i<d;i++){
        float val=0.0f;
        for(int j=0;j<n;j++){
            val+=w[i*n+j]*x[j];
        }
        out[i]=val;
    }
}
void matmul(float* out,float* x,BlockQ8*w,int n,int d){
    int num_blocks=n/Q_BLOCK_SIZE;
    // w (d x n) @ x(n x 1) -> out(d x 1)
    #pragma omp parallel for
    for(int i=0;i<d;i++){
        float val=0.0f;
        __m256 sum_vec=_mm256_setzero_ps();
        for(int b=0;b< num_blocks;b++){
            BlockQ8* curr_block=&w[i*num_blocks+b];

            float block_scale=curr_block->scale;
            int x_offset=b*Q_BLOCK_SIZE;
            __m256 scale_vec=_mm256_set1_ps(curr_block->scale);
            for(int chunk=0;chunk<4;chunk++){
                int off=chunk*8;
                __m128i int8_vals= _mm_loadl_epi64((__m128i const*)(curr_block->weights+off));
                __m256i int32_vals=_mm256_cvtepi8_epi32(int8_vals);
                __m256 float_weights=_mm256_cvtepi32_ps(int32_vals);

                __m256 raw_weights=_mm256_mul_ps(float_weights,scale_vec);
                __m256 x_vec= _mm256_loadu_ps(&x[x_offset+off]);
                sum_vec=_mm256_fmadd_ps(raw_weights,x_vec,sum_vec);
            }
        }
        __m128 vlow=_mm256_castps256_ps128(sum_vec);
        __m128 vhigh=_mm256_extractf128_ps(sum_vec,1);
        vlow=_mm_add_ps(vlow,vhigh);
        __m128 _shuffle=_mm_movehdup_ps(vlow);
        __m128 sums= _mm_add_ps(vlow,_shuffle);
        _shuffle=_mm_movehl_ps(_shuffle,sums);
        sums=_mm_add_ss(sums,_shuffle);
        out[i]=_mm_cvtss_f32(sums);
    }
}


void rms_norm(float* out,float*x,float* weight,int size){

    float sum_of_squares=0.0f;
    for(int i=0;i< size;i++){
        sum_of_squares+=x[i]*x[i];
    }
    sum_of_squares/=size;
    sum_of_squares+=1e-5f;
    sum_of_squares=1.0f/sqrtf(sum_of_squares);
    for(int i=0;i<size;i++){
        out[i]=weight[i]*(sum_of_squares*x[i]);
    }
}

void softmax(float* x, int size){
    float max_val=INT32_MIN;
    for(int i=0;i<size;i++){
        if(x[i]>max_val){
            max_val=x[i];
        }
    }

    float sum=0.0f;
    for (int i=0;i < size;i++){
        x[i]=expf(x[i]-max_val);
        sum+=x[i];
    }
    for(int i=0;i<size;i++){
        x[i]/=sum;
    }

}
float* forward(int token, int pos, Config_Transformer* c,Transformer_Weights* w, RunState*s){
    float* embedding=&(w->token_embedding_table[token * c->dim]);
    memcpy(s->x,embedding,c->dim* sizeof(float));

    int head_size=c->dim/c->n_heads;
    int kv_dim=(c->dim*c->n_kv_heads)/c->n_heads;
    int kv_mul=c->n_heads/c->n_kv_heads;
    for(int i=0;i< c->n_layers;i++){

        rms_norm(s->xb,s->x,w->rms_attn_weights+i*c->dim,c->dim);

        matmul(s->q,s->xb,w->wq+i*c->dim*c->dim/Q_BLOCK_SIZE,c->dim,c->dim);
        matmul(s->k,s->xb,w->wk+i*c->dim*kv_dim/Q_BLOCK_SIZE,c->dim,kv_dim);
        matmul(s->v,s->xb,w->wv+i*c->dim*kv_dim/Q_BLOCK_SIZE,c->dim,kv_dim);

        //RoPE
        for(int j=0;j<c->dim;j+=2){
            int head_dim=j%head_size;
            float fcr=w->freq_cis_real[pos*(head_size/2)+(head_dim)/2];
            float fci=w->freq_cis_imag[pos* (head_size/2)+head_dim/2];
            float q0=s->q[j];float q1=s->q[j+1];
            s->q[j]=q0*fcr-q1*fci;
            s->q[j+1]=q0*fcr+q1*fci;
            if(j<kv_dim){
                float k0=s->k[j];float k1=s->k[j+1];
                s->k[j]=k0*fcr-k1*fci;
                s->k[j+1]=k0*fcr+k1*fci;
            }
        }
        int offset=i*c->seq_len*kv_dim;
        memcpy(s->key_cache+offset+pos*kv_dim,s->k,kv_dim*sizeof(float));
        memcpy(s->value_cache+offset+pos*kv_dim,s->v,kv_dim*sizeof(float));

        // Multi Head Attention  //
        # pragma omp parallel for
        for (int h=0;h<c->n_heads;h++){
            float* q=s->q+h*head_size;
            float* att=s->attn+h*c->seq_len;

            int kv_h=h/kv_mul;

            for(int t=0;t<=pos;t++){
                float* k=s->key_cache+offset+t*kv_dim+kv_h*head_size;
                float score=0.0f;
                for(int p=0;p<head_size;p++){
                    score+=q[p]*k[p];
                }
                score/=sqrtf(head_size);
                att[t]=score;
            }
            softmax(att,pos+1);

            float*xb=s->xb+h*head_size;
            for(int p=0;p<head_size;p++) xb[p]=0.0f;

            for(int t=0;t<=pos;t++){
                float*v =s->value_cache+offset+t* kv_dim+kv_h*head_size;
                float a=att[t];
                for(int p=0;p<head_size;p++){
                    xb[p]+=a*v[p];
                }
            }
        }
        matmul(s->xb2,s->xb,w->wo+i*c->dim*c->dim/Q_BLOCK_SIZE,c->dim,c->dim);
        for(int p=0;p<c->dim;p++){s->x[p]+=s->xb2[p];}

        rms_norm(s->xb,s->x,w->rms_ffn_weight+i*c->dim,c->dim);
        matmul(s->hb,s->xb,w->w1+i*c->dim*c->hidden_dim/Q_BLOCK_SIZE,c->dim,c->hidden_dim);
        matmul(s->hb2,s->xb,w->w3+i*c->dim*c->hidden_dim/Q_BLOCK_SIZE,c->dim,c->hidden_dim);

        for(int p=0;p<c->hidden_dim;p++){
            float val=s->hb[p];
            val*=(1.0f/(1.0f+expf(-val)));
            val*=s->hb2[p];
            s->hb[p]=val;
        }
        matmul(s->xb,s->hb,w->w2+i*c->hidden_dim*c->dim/Q_BLOCK_SIZE,c->hidden_dim,c->dim);
        for(int p=0;p<c->dim;p++){s->x[p]+=s->xb[p];}
    }

    rms_norm(s->x,s->x,w->rms_final_weight,c->dim);
    matmul(s->logits,s->x,w->wcls,c->dim,c->vocab_size);
    return s->logits;
}
float* forward_paged(int token, int pos, Config_Transformer* c,Transformer_Weights* w, RunState*s,Sequence& seq, KVBlock_Manager& mem_manager,PrefixCacheManager& cache_mgr){
    
    seq.allocate_next_token(token,mem_manager,cache_mgr);
    float* embedding=&(w->token_embedding_table[token * c->dim]);
    memcpy(s->x,embedding,c->dim* sizeof(float));

    int head_size=c->dim/c->n_heads;
    int kv_dim=(c->dim*c->n_kv_heads)/c->n_heads;
    int kv_mul=c->n_heads/c->n_kv_heads;
    for(int i=0;i< c->n_layers;i++){

        rms_norm(s->xb,s->x,w->rms_attn_weights+i*c->dim,c->dim);

        matmul(s->q,s->xb,w->wq+i*c->dim*c->dim/Q_BLOCK_SIZE,c->dim,c->dim);
        matmul(s->k,s->xb,w->wk+i*c->dim*kv_dim/Q_BLOCK_SIZE,c->dim,kv_dim);
        matmul(s->v,s->xb,w->wv+i*c->dim*kv_dim/Q_BLOCK_SIZE,c->dim,kv_dim);

        //RoPE
        for(int j=0;j<c->dim;j+=2){
            int head_dim=j%head_size;
            float fcr=w->freq_cis_real[pos*(head_size/2)+(head_dim)/2];
            float fci=w->freq_cis_imag[pos* (head_size/2)+head_dim/2];
            float q0=s->q[j];float q1=s->q[j+1];
            s->q[j]=q0*fcr-q1*fci;
            s->q[j+1]=q0*fci+q1*fcr;
            if(j<kv_dim){
                float k0=s->k[j];float k1=s->k[j+1];
                s->k[j]=k0*fcr-k1*fci;
                s->k[j+1]=k0*fci+k1*fcr;
            }
        }
        // int offset=i*c->seq_len*kv_dim;
        // memcpy(s->key_cache+offset+pos*kv_dim,s->k,kv_dim*sizeof(float));
        // memcpy(s->value_cache+offset+pos*kv_dim,s->v,kv_dim*sizeof(float)); 

        int curr_logical_page=pos>>4;
        int curr_physical_id=seq.block_table[curr_logical_page];
        int curr_block_offset=pos&15;

        KVBlock& curr_block=mem_manager.get_block(curr_physical_id);
        int layer_off=i*BLOCK_SIZE*kv_dim;
        int token_off=curr_block_offset*kv_dim;
        memcpy(curr_block.k_ptr+layer_off+token_off,s->k,kv_dim*sizeof(float));
        memcpy(curr_block.v_ptr+layer_off+token_off,s->v,kv_dim*sizeof(float));


        // Multi Head Attention  //
        # pragma omp parallel for
        for (int h=0;h<c->n_heads;h++){
            float* q=s->q+h*head_size;
            float* att=s->attn+h*c->seq_len;
            int kv_h=h/kv_mul;
            int num_full_blocks=pos>>4;
            int curr_token_off=pos&15;
            for(int b=0;b<=num_full_blocks;b++){
                int t_physical_id= seq.block_table[b];
                KVBlock& target_block=mem_manager.get_block(t_physical_id);
                int end_t_offset=(b == num_full_blocks)? (pos &15):15;

                float* k_base= target_block.k_ptr+layer_off+(kv_h*head_size);
                for(int t_off=0;t_off<=end_t_offset;t_off++){
                   float*k=k_base+(t_off*kv_dim);
                   __m256 sum_vec=_mm256_setzero_ps();
                   for(int p=0;p<head_size;p+=8){
                    __m256 vq=_mm256_loadu_ps(&q[p]);
                    __m256 vk= _mm256_loadu_ps(&k[p]);
                    sum_vec=_mm256_fmadd_ps(vq,vk,sum_vec);
                   }

                   __m128 vlow=_mm256_castps256_ps128(sum_vec);
                    __m128 vhigh=_mm256_extractf128_ps(sum_vec,1);
                    vlow=_mm_add_ps(vlow,vhigh);
                    __m128 _shuffle=_mm_movehdup_ps(vlow);
                    __m128 sums= _mm_add_ps(vlow,_shuffle);
                    _shuffle=_mm_movehl_ps(_shuffle,sums);
                    sums=_mm_add_ss(sums,_shuffle);
                    float raw_score=_mm_cvtss_f32(sums);
                    att[b*BLOCK_SIZE+t_off]=raw_score/sqrtf(head_size);
                }
            }
            softmax(att,pos+1);

            float*xb_head=s->xb+h*head_size;
            for(int p=0;p<head_size;p++) xb_head[p]=0.0f;

             for(int b=0;b<=num_full_blocks;b++){
                int t_physical_id= seq.block_table[b];
                KVBlock& target_block=mem_manager.get_block(t_physical_id);
                int end_t_offset=(b == num_full_blocks)? (pos &15):15;
                float* v_base=target_block.v_ptr+layer_off+(kv_h*head_size);
                for(int t_off=0;t_off<=end_t_offset;t_off++){
                    float* v=v_base+(t_off*kv_dim);
                    float a=att[b*BLOCK_SIZE+t_off];
                    __m256 va= _mm256_set1_ps(a);
                    for(int p=0;p<head_size;p+=8){
                        __m256 v_xb=_mm256_loadu_ps(&xb_head[p]);
                        __m256 vv =_mm256_loadu_ps(&v[p]);
                        v_xb=_mm256_fmadd_ps(va,vv,v_xb);
                        _mm256_storeu_ps(&xb_head[p],v_xb);
                    }

                }
            }
        }
        matmul(s->xb2,s->xb,w->wo+i*c->dim*c->dim/Q_BLOCK_SIZE,c->dim,c->dim);
        for(int p=0;p<c->dim;p++){s->x[p]+=s->xb2[p];}

        rms_norm(s->xb,s->x,w->rms_ffn_weight+i*c->dim,c->dim);
        matmul(s->hb,s->xb,w->w1+i*c->dim*c->hidden_dim/Q_BLOCK_SIZE,c->dim,c->hidden_dim);
        matmul(s->hb2,s->xb,w->w3+i*c->dim*c->hidden_dim/Q_BLOCK_SIZE,c->dim,c->hidden_dim);

        for(int p=0;p<c->hidden_dim;p++){
            float val=s->hb[p];
            val*=(1.0f/(1.0f+expf(-val)));
            val*=s->hb2[p];
            s->hb[p]=val;
        }
        matmul(s->xb,s->hb,w->w2+i*c->hidden_dim*c->dim/Q_BLOCK_SIZE,c->hidden_dim,c->dim);
        for(int p=0;p<c->dim;p++){s->x[p]+=s->xb[p];}
    }

    rms_norm(s->x,s->x,w->rms_final_weight,c->dim);
    matmul(s->logits,s->x,w->wcls,c->dim,c->vocab_size);
    return s->logits;
}

void matmul_bulk(float* out, float*x,float* w, int batch, int n, int d){
    #pragma omp parallel for collapse(2)
    for( int b=0;b< batch;b++){
        for (int i=0;i < d;i++){
            float val=0.0f;
            for(int j=0;j< n;j++){
                val+=x[b*n+j]* w[i*n+j];
            }
            out[b*d+i]=val;
        }
    }
}
void matmul_bulk(float* out, float*x,BlockQ8* w, int batch, int n, int d){
    int num_blocks=n/Q_BLOCK_SIZE;
    // w (d x n) @ x(n x 1) -> out(d x 1)
    #pragma omp parallel for
    for(int i=0;i<d;i++){
        for(int b=0;b<batch;b++){
            out[b*d+i]=0.0f;
        }
        for(int b=0;b< num_blocks;b++){
            BlockQ8* curr_block=&w[i*num_blocks+b];
            int x_offset=b*Q_BLOCK_SIZE;
            __m256 scale_vec=_mm256_set1_ps(curr_block->scale);
            __m256 w_float[4];
            for(int chunk=0;chunk<4;chunk++){
                int off=chunk*8;
                __m128i int8_vals= _mm_loadl_epi64((__m128i const*)(curr_block->weights+off));
                __m256i int32_vals=_mm256_cvtepi8_epi32(int8_vals);
                __m256 float_weights=_mm256_cvtepi32_ps(int32_vals);
                w_float[chunk]=_mm256_mul_ps(float_weights,scale_vec);
    }
    for(int b=0;b<batch;b++){
        __m256 sum_vec=_mm256_setzero_ps();
        int x_token_off=(b*n)+x_offset;       
            for(int chunk=0;chunk<4;chunk++){
                __m256 x_vec= _mm256_loadu_ps(&x[x_token_off+chunk*8]);
                sum_vec=_mm256_fmadd_ps(w_float[chunk],x_vec,sum_vec);
            }
            __m128 vlow=_mm256_castps256_ps128(sum_vec);
            __m128 vhigh=_mm256_extractf128_ps(sum_vec,1);
            vlow=_mm_add_ps(vlow,vhigh);
            __m128 _shuffle=_mm_movehdup_ps(vlow);
            __m128 sums= _mm_add_ps(vlow,_shuffle);
            _shuffle=_mm_movehl_ps(_shuffle,sums);
            sums=_mm_add_ss(sums,_shuffle);
            out[b*d+i]=_mm_cvtss_f32(sums);
        }
    }
    }
}



float* forward_prefill_chunk(const vector<int>& tokens, int start_pos,Config_Transformer*c,Transformer_Weights* w,RunState*s,Sequence& seq,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr){
    int num_tokens= tokens.size();
    if(num_tokens ==0)return nullptr;

    for( int i=0;i < num_tokens;i++){
        seq.allocate_next_token(tokens[i],mem_mgr,cache_mgr);
    }
    #pragma omp parallel for
    for( int t=0;t<num_tokens;t++){
        float* embedding=&(w->token_embedding_table[tokens[t]* c->dim]);
        memcpy(s->x+(t*c->dim),embedding,c->dim*sizeof(float));
    }
    int head_size=c->dim/c->n_heads;
    int kv_dim=(c->dim*c->n_kv_heads)/c->n_heads;
    int kv_mul=c->n_heads/c->n_kv_heads;


    for(int l=0;l<c->n_layers;l++){

        #pragma omp parallel for
        for(int t=0;t< num_tokens;t++){
            rms_norm(s->xb+(t* c->dim),s->x+(t*c->dim),w->rms_attn_weights+l*c->dim,c->dim);
        }
        matmul_bulk(s->q,s->xb,w->wq+l*c->dim*c->dim/Q_BLOCK_SIZE,num_tokens,c->dim,c->dim);
        matmul_bulk(s->k,s->xb,w->wk+l*c->dim*kv_dim/Q_BLOCK_SIZE,num_tokens,c->dim,kv_dim);
        matmul_bulk(s->v,s->xb,w->wv+l*c->dim*kv_dim/Q_BLOCK_SIZE,num_tokens,c->dim,kv_dim);

        #pragma omp parallel for
        for(int t=0;t<num_tokens;t++){
            int pos=start_pos+t;
            float* qt= s->q+(t* c->dim);
            float* kt=s->k+(t*kv_dim);

            for( int j=0;j<c->dim;j+=2){
                int head_dim=j%head_size;
                float fcr= w->freq_cis_real[pos*(head_size/2)+(head_dim/2)];
                float fci=w->freq_cis_imag[pos*(head_size/2)+(head_dim/2)];

                float q0=qt[j];float q1=qt[j+1];
                qt[j]=q0*(fcr)-q1* fci;
                qt[j+1]=q0*fci+q1*fcr;

                if(j<kv_dim){
                    float k0=kt[j];float k1=kt[j+1];
                    kt[j]=k0*fcr-k1*fci;
                    kt[j+1]=k0*fci+k1*fcr;
                }
            }
        }
        int layer_off=l*BLOCK_SIZE*kv_dim;

        #pragma omp parallel for
        for(int t=0;t<num_tokens;t++){
            int pos=start_pos+t;
            int logical_page=pos/BLOCK_SIZE;
            int phy_id=seq.block_table[logical_page];
            int block_off=pos%BLOCK_SIZE;
            KVBlock& target_block=mem_mgr.get_block(phy_id);
            memcpy(target_block.k_ptr+layer_off+(block_off*kv_dim),s->k+(t*kv_dim),kv_dim*(sizeof(float)));
            memcpy(target_block.v_ptr+layer_off+(block_off*kv_dim),s->v+(t*kv_dim),kv_dim*sizeof(float));
        }
        #pragma omp parallel for collapse(2)
        for(int t=0;t< num_tokens;t++){
            for(int h=0;h< c->n_heads;h++){
                int pos=start_pos+t;
                float* q=s->q+(t*c->dim)+(h* head_size);
                float* xb_out=s->xb+(t*c->dim)+(h* head_size);
                int kv_h=h/kv_mul;
                for(int p=0;p<head_size;p++) xb_out[p]=0.0f;
                float m_global=-1e9f;
                float d_global=0.0f;
                int num_full_blocks=pos/BLOCK_SIZE;
                for(int b=0;b<=num_full_blocks;b++){
                    int phy_id=seq.block_table[b];
                    KVBlock& target_block=mem_mgr.get_block(phy_id);
                    int end_t_off=(b==num_full_blocks)? (pos% BLOCK_SIZE):(BLOCK_SIZE-1);
                    float* k_base=target_block.k_ptr+layer_off+(kv_h*head_size);
                    float* v_base=target_block.v_ptr+layer_off+(kv_h*head_size);

                    for(int t_off=0;t_off<=end_t_off;t_off++){
                        float* k=k_base+(t_off*kv_dim);
                        float raw_score=0.0f;
                        for(int p=0;p<head_size;p++)raw_score+=q[p]*k[p];
                        raw_score/=sqrtf(head_size);
                        if(raw_score> m_global){
                            float correction=expf(m_global-raw_score);
                            for(int p=0;p<head_size;p++)xb_out[p]*=correction;
                            d_global*=correction;m_global=raw_score;
                        }
                        float exp_score=expf(raw_score-m_global);
                        d_global+=exp_score;
                        float* v=v_base+(t_off*kv_dim);
                        for(int p=0;p<head_size;p++){
                            xb_out[p]+=exp_score*v[p];
                        }
                    }
                }
                for(int p=0;p<head_size;p++)xb_out[p]/=d_global;
            }
        }
        matmul_bulk(s->xb2,s->xb,w->wo+l*c->dim*c->dim/Q_BLOCK_SIZE,num_tokens,c->dim,c->dim);
        #pragma omp parallel for
        for(int t=0;t<num_tokens;t++){
            for(int i=0;i<c->dim;i++) s->x[t*c->dim+i]+=s->xb2[t*c->dim+i];
        }
         #pragma omp parallel for
        for(int t=0;t<num_tokens;t++){
        rms_norm(s->xb+t*c->dim,s->x+t*c->dim,w->rms_ffn_weight+l*c->dim,c->dim);
        }
        matmul_bulk(s->hb,s->xb,w->w1+l*c->dim*c->hidden_dim/Q_BLOCK_SIZE,num_tokens,c->dim,c->hidden_dim);
       matmul_bulk(s->hb2,s->xb,w->w3+l*c->dim*c->hidden_dim/Q_BLOCK_SIZE,num_tokens,c->dim,c->hidden_dim);
       #pragma omp parallel for
       for(int t=0;t<num_tokens;t++){
        float* hb=s->hb+t*c->hidden_dim;
        float* hb2=s->hb2+t*c->hidden_dim;
        for(int i=0;i<c->hidden_dim;i++){
            float val=hb[i];
            val*=(1.0f/(1.0f+expf(-val)));
            val*=hb2[i];
            hb[i]=val;
        }
       } 
       matmul_bulk(s->xb,s->hb,w->w2+l* c->hidden_dim*c->dim/Q_BLOCK_SIZE,num_tokens,c->hidden_dim,c->dim);
       #pragma omp parallel for
       for(int t=0;t< num_tokens;t++){
        for(int i=0;i<c->dim;i++) s->x[t*c->dim+i]+=s->xb[t*c->dim+i];

       }
    }
    float* last_token_x=s->x+(num_tokens-1)*c->dim;
    rms_norm(s->xb,last_token_x,w->rms_final_weight,c->dim);
    matmul_bulk(s->logits,s->xb,w->wcls,1,c->dim,c->vocab_size);
    return s->logits;
}

int argmax(float* prob, int size){
    int max_idx=0;
    float max_prob=0.0f;
    for(int i=0;i<size;i++){
        if(prob[i] > max_prob){
            max_idx=i;
            max_prob=prob[i];
        }
    }
    return max_idx;
}

float* forward_decode_batch(const vector<int>& batch_tokens, const vector<int>& batch_positions,vector<Request*>& batch_reqs,Config_Transformer*c,Transformer_Weights* w,RunState*s,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr){
    int B= batch_tokens.size();
    if(B ==0)return nullptr;
    int head_size=c->dim/c->n_heads;
    int kv_dim=(c->dim*c->n_kv_heads)/c->n_heads;
    int kv_mul=c->n_heads/c->n_kv_heads;

    #pragma omp parallel for
    for(int b=0;b<B;b++){
        batch_reqs[b]->seq.allocate_next_token(batch_tokens[b],mem_mgr,cache_mgr);
        memcpy(s->x+(b*c->dim),&w->token_embedding_table[batch_tokens[b]*c->dim],c->dim*sizeof(float));
    }


    for(int l=0;l<c->n_layers;l++){
        #pragma omp parallel for
        for(int b=0;b<B;b++){
            rms_norm(s->xb+(b* c->dim),s->x+(b*c->dim),w->rms_attn_weights+l*c->dim,c->dim);
        }
        matmul_bulk(s->q,s->xb,w->wq+l*c->dim*c->dim/Q_BLOCK_SIZE,B,c->dim,c->dim);
        matmul_bulk(s->k,s->xb,w->wk+l*c->dim*kv_dim/Q_BLOCK_SIZE,B,c->dim,kv_dim);
        matmul_bulk(s->v,s->xb,w->wv+l*c->dim*kv_dim/Q_BLOCK_SIZE,B,c->dim,kv_dim);

        #pragma omp parallel for
        for(int b=0;b<B;b++){
            int pos=batch_positions[b];
            float* qb= s->q+(b* c->dim);
            float* kb=s->k+(b*kv_dim);
            float* vb=s->v +(b*kv_dim);

            for( int j=0;j<c->dim;j+=2){
                int head_dim=j%head_size;
                float fcr= w->freq_cis_real[pos*(head_size/2)+(head_dim/2)];
                float fci=w->freq_cis_imag[pos*(head_size/2)+(head_dim/2)];

                float q0=qb[j];float q1=qb[j+1];
                qb[j]=q0*(fcr)-q1* fci;
                qb[j+1]=q0*fci+q1*fcr;

                if(j<kv_dim){
                    float k0=kb[j];float k1=kb[j+1];
                    kb[j]=k0*fcr-k1*fci;
                    kb[j+1]=k0*fci+k1*fcr;
                }
            }
            Sequence& seq=batch_reqs[b]->seq;
            int curr_page=pos/BLOCK_SIZE;
            int phy_id=seq.block_table[curr_page];
            int block_off= pos% BLOCK_SIZE;
            int layer_off=l*BLOCK_SIZE*kv_dim;
            KVBlock& target_block=mem_mgr.get_block(phy_id);
            memcpy(target_block.k_ptr+layer_off+(block_off*kv_dim),kb,kv_dim*sizeof(float));
            memcpy(target_block.v_ptr+layer_off+(block_off* kv_dim),vb,kv_dim*sizeof(float));
        }
        #pragma omp parallel for collapse(2)
        for(int b=0;b<B;b++){
            for(int h=0;h< c->n_heads;h++){
                int pos=batch_positions[b];
                Sequence& seq=batch_reqs[b]->seq;
                float* q=s->q+(b*c->dim)+(h* head_size);
                float* att=s->attn+(b* c->n_heads*c->seq_len)+(h* c->seq_len);
                int kv_h=h/kv_mul;
                int layer_off=l*BLOCK_SIZE*kv_dim;
                int num_full_blocks=pos/BLOCK_SIZE;
                for(int blk=0;blk<=num_full_blocks;blk++){
                    int phy_id=seq.block_table[blk];
                    KVBlock& target_block=mem_mgr.get_block(phy_id);
                    int end_t_off=(blk==num_full_blocks)? (pos% BLOCK_SIZE):(BLOCK_SIZE-1);
                    float* k_base=target_block.k_ptr+layer_off+(kv_h*head_size);
                    for(int t_off=0;t_off<=end_t_off;t_off++){
                        float* k=k_base+(t_off*kv_dim);
                        float raw_score=0.0f;
                        for(int p=0;p<head_size;p++)raw_score+=q[p]*k[p];
                        att[blk*BLOCK_SIZE+t_off]=raw_score/sqrtf(head_size);
                    }
                }
                softmax(att,pos+1);
                float* xb_out=s->xb+(b*c->dim)+(h* head_size);
                for(int p=0;p<head_size;p++)xb_out[p]=0.0f;

                for(int blk=0;blk<=num_full_blocks;blk++){
                    int phy_id=seq.block_table[blk];
                    float* v_base=mem_mgr.get_block(phy_id).v_ptr+layer_off+(kv_h* head_size);
                    int end_t_offset=(blk==num_full_blocks)? (pos% BLOCK_SIZE): (BLOCK_SIZE -1);
                    for(int t_off=0;t_off<= end_t_offset;t_off++){
                        float* v=v_base+(t_off*kv_dim);
                        float a=att[blk*BLOCK_SIZE+t_off];
                        for(int p=0;p<head_size;p++) xb_out[p]+=a*v[p];
                    }
                }
            }
        }
        matmul_bulk(s->xb2,s->xb,w->wo+l*c->dim*c->dim/Q_BLOCK_SIZE,B,c->dim,c->dim);
        #pragma omp parallel for
        for(int b=0;b<B;b++){
            for(int i=0;i<c->dim;i++) s->x[b*c->dim+i]+=s->xb2[b*c->dim+i];

            rms_norm(s->xb+b*c->dim,s->x+b*c->dim,w->rms_ffn_weight+l*c->dim,c->dim);
        }
        matmul_bulk(s->hb,s->xb,w->w1+l*c->dim*c->hidden_dim/Q_BLOCK_SIZE,B,c->dim,c->hidden_dim);
       matmul_bulk(s->hb2,s->xb,w->w3+l*c->dim*c->hidden_dim/Q_BLOCK_SIZE,B,c->dim,c->hidden_dim);
       #pragma omp parallel for
       for(int b=0;b<B;b++){
        float* hb=s->hb+b*c->hidden_dim;
        float* hb2=s->hb2+b*c->hidden_dim;
        for(int i=0;i<c->hidden_dim;i++){
            float val=hb[i];
            val*=(1.0f/(1.0f+expf(-val)));
            hb[i]=val* hb2[i];
        }
       } 
       matmul_bulk(s->xb,s->hb,w->w2+l* c->hidden_dim*c->dim/Q_BLOCK_SIZE,B,c->hidden_dim,c->dim);
       #pragma omp parallel for
       for(int b=0;b< B;b++){
        for(int i=0;i<c->dim;i++) s->x[b*c->dim+i]+=s->xb[b*c->dim+i];
    }
    }
    #pragma omp parallel for
       for(int b=0;b< B;b++){
        rms_norm(s->x+b*c->dim,s->x+b*c->dim,w->rms_final_weight,c->dim);
       }
    matmul_bulk(s->logits,s->x,(float*)w->wcls,B,c->dim,c->vocab_size);
    return s->logits;
}
int ingest_prompt(const vector<int>& prompt_tokens,Sequence& seq,KVBlock_Manager& mem_mgr,PrefixCacheManager& cache_mgr,Config_Transformer*c,Transformer_Weights* w,RunState*s,bool enable_prefix_caching){
    int p=0;vector<int> tokens_to_compute;
    // cout<<"Checking in prefix cache"<<endl;
    if(enable_prefix_caching){
        while(p+BLOCK_SIZE<=prompt_tokens.size()){
            vector<int> chunk(prompt_tokens.begin()+p,prompt_tokens.begin()+p+BLOCK_SIZE);
            uint64_t hash=hash_block_state(seq.current_hash,chunk);
            int hit_id=cache_mgr.check_cache(hash);
            if(hit_id!=-1){
                mem_mgr.inc_ref_count(hit_id,cache_mgr);
                seq.block_table.push_back(hit_id);
                seq.logical_token_count+=BLOCK_SIZE;
                seq.current_hash=hash;
                // cout<<"Cache Hit for tokens: "<<p<<" - "<<p+BLOCK_SIZE-1<<" Mapped to block "<<hit_id<<endl;
                p+=BLOCK_SIZE;
            }
            else{
                break;
            }    
        }
    }
        while(p < prompt_tokens.size()){
            tokens_to_compute.push_back(prompt_tokens[p]);p++;
        }
        if(!tokens_to_compute.empty()){
            forward_prefill_chunk(tokens_to_compute,seq.logical_token_count,c,w,s,seq,mem_mgr,cache_mgr);  
        }

        return seq.logical_token_count;
    }

int main() {
    cout << "\n=================================================================" << endl;
    cout << "     NITRORAG: SCIENTIFIC HARDWARE BENCHMARK SUITE (REAL DATA)   " << endl;
    cout << "=================================================================\n" << endl;

    // -----------------------------------------------------------------------
    // 1. HARDWARE & MODEL SETUP (INT8 Q8_0)
    // -----------------------------------------------------------------------
    int fd = open("stories110M_q8.bin", O_RDONLY);
    if(fd < 0) { cerr << "Open Failed. Missing stories110M_q8.bin" << endl; exit(1); }
    struct stat st; fstat(fd, &st);
    void* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    int* header = (int*)data;
    Config_Transformer c;
    c.dim = header[0]; c.hidden_dim = header[1]; c.n_layers = header[2];
    c.n_heads = header[3]; c.n_kv_heads = header[4]; c.vocab_size = header[5]; c.seq_len = header[6];

    int head_size = c.dim / c.n_heads;
    int kv_dim = c.n_kv_heads * head_size;
    int Q_BLOCK_SIZE = 32;

    uint8_t* byte_ptr = (uint8_t*)data + 28;
    Transformer_Weights w;
    
    w.token_embedding_table = (float*)byte_ptr; byte_ptr += c.vocab_size * c.dim * sizeof(float);
    w.rms_attn_weights = (float*)byte_ptr;      byte_ptr += c.n_layers * c.dim * sizeof(float);
    
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

    // -----------------------------------------------------------------------
    // 2. STATE ALLOCATION
    // -----------------------------------------------------------------------
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

    int TOTAL_BLOCKS = 8000; 
    double kb_per_token = (2.0 * c.n_layers * kv_dim * sizeof(float)) / 1024.0;
    vector<int> shared_prompt(256, 263); 

    // =======================================================================
    // EXPERIMENT 1: PAGED MEMORY & FRAGMENTATION
    // =======================================================================
    cout << "\n[Running Exp 1: Memory Scaling & Fragmentation...]\n" << endl;
    printf("| Users | Naive Alloc (MB) | Paged Used (MB) | Paged Alloc (MB) | Fragmentation |\n");
    printf("|-------|-----------------:|----------------:|-----------------:|--------------:|\n");

    vector<int> user_counts = {1, 4, 16, 32, 64};
    for (int users : user_counts) {
        KVBlock_Manager mem(TOTAL_BLOCKS, c.n_layers, kv_dim);
        PrefixCacheManager cache(mem, TOTAL_BLOCKS);
        
        vector<Sequence*> seqs;
        int total_logical_tokens = 0;

        for (int i = 0; i < users; i++) {
            Sequence* sq = new Sequence(i);
            int prompt_len = 100 + (i * 5); 
            vector<int> p(prompt_len, 263);
            
            ingest_prompt(p, *sq, mem, cache, &c, &w, &s, false); // Cache OFF to measure raw allocation
            seqs.push_back(sq);
            total_logical_tokens += prompt_len;
        }

        double naive_mb = (users * 4096 * kb_per_token) / 1024.0; 
        int active_blocks = TOTAL_BLOCKS - mem.get_free_count();
        double paged_alloc_mb = (active_blocks * BLOCK_SIZE * kb_per_token) / 1024.0;
        double paged_used_mb = (total_logical_tokens * kb_per_token) / 1024.0;
        double fragmentation_mb = paged_alloc_mb - paged_used_mb;

        if (naive_mb > 8192.0) {
            printf("| %-5d | %-16s | %-15.2f | %-16.2f | %-13.2f |\n", 
                   users, "OOM (>8GB)", paged_used_mb, paged_alloc_mb, fragmentation_mb);
        } else {
            printf("| %-5d | %-16.2f | %-15.2f | %-16.2f | %-13.2f |\n", 
                   users, naive_mb, paged_used_mb, paged_alloc_mb, fragmentation_mb);
        }

        for (auto sq : seqs) { sq->release(mem, cache); delete sq; }
    }

    // =======================================================================
    // EXPERIMENT 2: OPENMP THREAD SCALING
    // =======================================================================
    cout << "\n[Running Exp 2: OpenMP CPU Scaling (Decode)]\n" << endl;
    printf("| Threads | Decode Tokens/sec |\n");
    printf("|---------|------------------:|\n");

    vector<int> thread_counts = {1, 2, 4, 8,10,12,14,16};
    for (int t : thread_counts) {
        omp_set_num_threads(t); 
        
        KVBlock_Manager mem(1000, c.n_layers, kv_dim);
        PrefixCacheManager cache(mem, 1000);
        Sequence sq(1);
        int pos = ingest_prompt(shared_prompt, sq, mem, cache, &c, &w, &s, false);
        
        int toks_to_gen = 50;
        auto t_start = high_resolution_clock::now();
        for (int i = 0; i < toks_to_gen; i++) {
            forward_paged(100, pos + i, &c, &w, &s, sq, mem, cache);
        }
        auto t_end = high_resolution_clock::now();
        
        double elapsed_sec = duration<double>(t_end - t_start).count();
        double tps = toks_to_gen / elapsed_sec;
        
        printf("| %-7d | %-17.2f |\n", t, tps);
        sq.release(mem, cache);
    }
    omp_set_num_threads(omp_get_num_procs()); 

    // =======================================================================
    // EXPERIMENT 3: CONCURRENT BATCHING SCALABILITY
    // =======================================================================
    cout << "\n[Running Exp 3: Continuous Batching Throughput]\n" << endl;
    printf("| Concurrent Users | Aggregate Throughput (tok/s) | Avg TTFT (ms) |\n");
    printf("|------------------|-----------------------------:|--------------:|\n");

    vector<int> batch_sizes = {1, 4, 8, 16, 32};

    for (int users : batch_sizes) {
        KVBlock_Manager mem(8000, c.n_layers, kv_dim);
        PrefixCacheManager cache(mem, 8000);
        Continuous_Scheduler sched; 
        
        vector<Request*> reqs;
        for (int i = 0; i < users; i++) {
            Request* req = new Request(i, shared_prompt);
            reqs.push_back(req);
            sched.add_request(req);
        }

        auto t_start = high_resolution_clock::now();
        double total_ttft = 0;
        int toks_generated = 0;
        unordered_map<Request*, bool> ttft_recorded; 

       while (sched.has_active_requests()) {
            sched.step(&c, &w, &s, mem, cache);
            for (auto req : reqs) {
                if (req->is_prefill_done && !ttft_recorded[req]) {
                    auto t_first_tok = high_resolution_clock::now();
                    total_ttft += duration<double, std::milli>(t_first_tok - t_start).count();
                    ttft_recorded[req] = true;
                }
            }
        }
        auto t_end = high_resolution_clock::now();
        
        double elapsed_sec = duration<double>(t_end - t_start).count();
        double aggregate_tps = (users * 20.0) / elapsed_sec; 
        double avg_ttft = total_ttft / users;

        printf("| %-16d | %-28.2f | %-13.2f |\n", users, aggregate_tps, avg_ttft);

        for (auto req : reqs) delete req;
    }

    // =======================================================================
    // EXPERIMENT 4: RADIX PREFIX CACHE EFFICIENCY
    // =======================================================================
    cout << "\n[Running Exp 4: Radix Prefix Cache Effectiveness]\n" << endl;
    printf("| Status     | Prefill Latency (ms) | Peak RAM Used (MB) |\n");
    printf("|------------|---------------------:|-------------------:|\n");

    KVBlock_Manager mem_rag(1000, c.n_layers, kv_dim);
    PrefixCacheManager cache_rag(mem_rag, 1000);

    // User A computes it normally
    Sequence seqA(1);
    auto ta1 = high_resolution_clock::now();
    ingest_prompt(shared_prompt, seqA, mem_rag, cache_rag, &c, &w, &s, true); // CACHE ON
    seqA.flush_cache(cache_rag);
    auto ta2 = high_resolution_clock::now();
    double time_miss = duration<double, std::milli>(ta2 - ta1).count();
    
    // User B shares it!
    Sequence seqB(2);
    auto tb1 = high_resolution_clock::now();
    ingest_prompt(shared_prompt, seqB, mem_rag, cache_rag, &c, &w, &s, true); // CACHE ON
    seqB.flush_cache(cache_rag);
    auto tb2 = high_resolution_clock::now();
    double time_hit = duration<double, std::milli>(tb2 - tb1).count();

    double ram_used = ((1000 - mem_rag.get_free_count()) * BLOCK_SIZE * kb_per_token) / 1024.0;

    printf("| %-10s | %-20.10f | %-18.2f |\n", "Cache Miss", time_miss, ram_used);
    printf("| %-10s | %-20.10f | %-18.2f |\n", "Cache Hit", time_hit, ram_used); 

    cout << "\n=================================================================" << endl;
    
    munmap(data, st.st_size);
    close(fd);
    return 0;
}