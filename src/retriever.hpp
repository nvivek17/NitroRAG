#pragma once
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stack>
#include <cctype>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <random>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

struct Document{
    int id;
    vector<float> embedding;
    Document(int ID): id(ID){}
    Document(int ID,vector<float> emb): id(ID),embedding(std::move(emb)){}
};

// struct MEMTABLE{
//     vector<Document> documents;
//     void insert(Document& doc){
//         documents.push_back(doc);
//     }
//     void insert(int id, vector<float>& embedding){
//         documents.emplace_back(id,embedding);
//     }
//     float consine_similarity(vector<float>& emb1,vector<float>& emb2){
//         if(emb1.size()!=emb2.size()){cout<<"Number of Dimensions doesn't match"<<endl;return INT32_MIN;}
//         int n_dimensions=emb1.size();float dot_prod=0.0f;float magnitude1=0.0f;float magnitude2=0.0f;
//         for(int i=0;i<n_dimensions;i++){
//             dot_prod+=emb1[i]*emb2[i];
//             magnitude1+=emb1[i]*emb1[i];
//             magnitude2+=emb2[i]*emb2[i];
//         }
//         if(magnitude1==0.0f||magnitude2==0.0f){return 0.0f;}
//         return dot_prod/(sqrt(magnitude1)*sqrt(magnitude2));

//     }
//     vector<int> search(vector<float>& query,int k=5){
//         priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> Q;
//         for(int i=0;i<documents.size();i++){
//             Q.push({consine_similarity(query,documents[i].embedding),documents[i].id});
//             while(Q.size()>k){
//                 Q.pop();
//             }
//         }
//         vector<int> results;
//         for(int i=0;i<k;i++){
//             int id=Q.top().second;Q.pop();
//             results.push_back(id);
//         }
//         std::reverse(results.begin(),results.end());
//         return results;
//     }
//     void flush_to_disk(const string& filename){
//         std::ofstream out(filename,std::ios::binary);
//         if(!out){
//             cerr<<"Error: Could not open the file for writing"<<endl;
//             return;
//         }
//         for(const auto& doc:documents){
//             out.write(reinterpret_cast<const char*>(&doc.id),sizeof(int));
//             int n_dimensions=doc.embedding.size();
//             out.write(reinterpret_cast<const char*>(doc.embedding.data()),n_dimensions*sizeof(float));
//         }
//         out.close();
//         cout<<"Flushed"<<documents.size()<<"vector to disk as "<< filename <<endl;
//         documents.clear();
//     }
// };
    struct MEMTABLE{
        vector<int> ids;
        vector<float> embeddings;
        int num_vectors=0;
        int n_dimensions=128;
        void reserve(int capacity){
            ids.reserve(capacity);
            embeddings.reserve(capacity*n_dimensions);
        }
        void clear(){
            ids.clear();
            embeddings.clear();
            num_vectors=0;
        }
        void insert(int id, const vector<float>& embedding){
            ids.push_back(id);
            embeddings.insert(embeddings.end(),embedding.begin(),embedding.end());
            num_vectors++;
        }
        float consine_similarity(const vector<float>& emb1,const float* emb2,int dim=128)const{
            if(emb1.size()!=dim){cout<<"Number of Dimensions doesn't match"<<endl;return INT32_MIN;}
            int n_dimensions=dim;float dot_prod=0.0f;float magnitude1=0.0f;float magnitude2=0.0f;
            for(int i=0;i<n_dimensions;i++){
                dot_prod+=emb1[i]*emb2[i];
                magnitude1+=emb1[i]*emb1[i];
                magnitude2+=emb2[i]*emb2[i];
            }
            if(magnitude1==0.0f||magnitude2==0.0f){return 0.0f;}
            return dot_prod/(sqrt(magnitude1)*sqrt(magnitude2));
        }
        float cosine_similarity_simd(const vector<float>& emb1,const float* emb2,int dim=128)const{
             if(emb1.size()!=dim){cout<<"Number of Dimensions doesn't match"<<endl;return INT32_MIN;}
            int n_dimensions=dim;float dot_prod=0.0f;float magnitude1=0.0f;float magnitude2=0.0f;
            __m256 simd_dot=_mm256_setzero_ps();
            __m256 simd_mag1=_mm256_setzero_ps();
            __m256 simd_mag2=_mm256_setzero_ps();
            for(int i=0;i<n_dimensions;i++){
                __m256 v1=_mm256_loadu_ps(&emb1[i]);
                __m256 v2=_mm256_loadu_ps(&emb2[i]);
                
                simd_dot=_mm256_fmadd_ps(v1,v2,simd_dot);
                simd_mag1=_mm256_fmadd_ps(v1,v1,simd_mag1);
                simd_mag2=_mm256_fmadd_ps(v2,v2,simd_mag2);
            }
            float dot_arr[8];float v1_arr[8];float v2_arr[8];
            _mm256_storeu_ps(dot_arr,simd_dot);
            _mm256_storeu_ps(v1_arr,simd_mag1);
            _mm256_storeu_ps(v2_arr,simd_mag2);
            for(int i=0;i<8;i++){
                dot_prod+=dot_arr[i];
                magnitude1+=v1_arr[i];
                magnitude2+=v2_arr[i];
            }
            __m256 t1=_mm256_hadd_ps(simd_dot,simd_dot);
            __m256 t2=_mm256_hadd_ps(t1,t1);
            __m128 t3=_mm256_extractf128_ps(t2,1);
            __m128 t4=_mm256_castps256_ps128(t2);
            __m128 t5=_mm_add_ps(t3,t4);
            dot_prod=_mm_cvtss_f32(t5);

            t1=_mm256_hadd_ps(simd_mag1,simd_mag1);
            t2=_mm256_hadd_ps(t1,t1);
            t3=_mm256_extractf128_ps(t2,1);
            t4=_mm256_castps256_ps128(t2);
            t5=_mm_add_ps(t3,t4);
            magnitude1=_mm_cvtss_f32(t5);

            t1=_mm256_hadd_ps(simd_mag2,simd_mag2);
            t2=_mm256_hadd_ps(t1,t1);
            t3=_mm256_extractf128_ps(t2,1);
            t4=_mm256_castps256_ps128(t2);
            t5=_mm_add_ps(t3,t4);
            magnitude2=_mm_cvtss_f32(t5);


            if(magnitude1==0.0f||magnitude2==0.0f){return 0.0f;}
            return dot_prod/(sqrt(magnitude1)*sqrt(magnitude2));
        }
        // vector<int> search(vector<float>& query,int k=5) const {
        //     priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> Q;
        //     for(int i=0;i<num_vectors;i++){
        //         Q.push({consine_similarity(query,&(embeddings[i*n_dimensions]),n_dimensions),ids[i]});
        //         while(Q.size()>k){
        //             Q.pop();
        //         }
        //     }
        //     vector<int> results;
        //     for(int i=0;i<k;i++){
        //         int id=Q.top().second;Q.pop();
        //         results.push_back(id);
        //     }
        //     std::reverse(results.begin(),results.end());
        //     return results;
        // }

    vector<int> search_parallel(const vector<float>& query, int k = 5) const {
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> global_Q;
        #pragma omp parallel
        {
            priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> Q;

            #pragma omp for nowait
            for(int i=0;i<num_vectors;i++){
                float score=consine_similarity(query,&embeddings[i*n_dimensions],n_dimensions);
                Q.push({score,ids[i]});
                while(Q.size()>k){
                        Q.pop();
                    }
            }
            #pragma omp critical
            {
                while(!Q.empty()){
                    global_Q.push(Q.top());Q.pop();
                    while(Q.size()>k){
                        Q.pop();
                    }
                }
            }
        }
        vector<int> results;
                for(int i=0;i<k;i++){
                    int id=global_Q.top().second;global_Q.pop();
                    results.push_back(id);
                }
                std::reverse(results.begin(),results.end());
                return results;
        
    }

    vector<int> search_parallel_simd(const vector<float>& query, int k = 5) const {
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> global_Q;
        #pragma omp parallel
        {
            priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> Q;

            #pragma omp for nowait
            for(int i=0;i<num_vectors;i++){
                float score=cosine_similarity_simd(query,&embeddings[i*n_dimensions],n_dimensions);
                Q.push({score,ids[i]});
                while(Q.size()>k){
                        Q.pop();
                    }
            }
            #pragma omp critical
            {
                while(!Q.empty()){
                    global_Q.push(Q.top());Q.pop();
                    while(Q.size()>k){
                        Q.pop();
                    }
                }
            }
        }
        vector<int> results;
                for(int i=0;i<k;i++){
                    int id=global_Q.top().second;global_Q.pop();
                    results.push_back(id);
                }
                std::reverse(results.begin(),results.end());
                return results;
    
    }
        void flush_to_disk(const string& filename){
            std::ofstream out(filename,std::ios::binary);
            if(!out){
                cerr<<"Error: Could not open the file for writing"<<endl;
                return;
            }
            out.write(reinterpret_cast<const char*>(&num_vectors),sizeof(int));
            out.write(reinterpret_cast<const char*>(&n_dimensions),sizeof(int));
            out.write(reinterpret_cast<const char*>(ids.data()),sizeof(int)*num_vectors);
            out.write(reinterpret_cast<const char*>(embeddings.data()),sizeof(float)*num_vectors*n_dimensions);
            // for(int i=0;i<num_vectors;i++){
            //     out.write(reinterpret_cast<const char*>(ids[i]),sizeof(int));
            //     out.write(reinterpret_cast<const char*>(&embeddings[i*n_dimensions]),n_dimensions*sizeof(float));
            // }
            out.close();
            cout<<"Flushed"<<num_vectors<<"vector to disk as "<< filename <<endl;
            clear();
        }

    };

struct SSTable{
    int fd;
    size_t file_size;
    void* mapped_data;
    int num_vectors;
    int n_dimensions;
    const int* ids;
    const float* embeddings;

    bool load(string filename){
        fd=open(filename.c_str(),O_RDONLY);
        if(fd<0){
            cerr<< "failed to open "<<filename<<endl;
            return false;
        }
        struct stat sb;
        if(fstat(fd,&sb)<0){
            cerr<<"failed to get file size"<<endl;
            close(fd);
            return false;
        }
        file_size=sb.st_size;
        
        mapped_data=mmap(nullptr,file_size,PROT_READ,MAP_PRIVATE,fd,0);
        if(mapped_data==MAP_FAILED){
            cerr<<"mmap failed"<<endl;
            close(fd);
            return false;
        }
        char* ptr=static_cast<char*>(mapped_data);
        num_vectors=*reinterpret_cast<const int*>(ptr);
        ptr+=sizeof(int);
        n_dimensions=*reinterpret_cast<const int*>(ptr);
        ptr+=sizeof(int);
        ids=reinterpret_cast<const int*>(ptr);
        ptr+=sizeof(int)*num_vectors;
        embeddings=reinterpret_cast<const float*>(ptr);
        cout<<"Mapping is successful, filename: "<<filename<<"| Num of vectors: "<<num_vectors<<"| Num of dimensions: "<<n_dimensions<<endl;
        return true;
    }
    void close_file(){
        if(mapped_data&& mapped_data!=MAP_FAILED){
            munmap(mapped_data,file_size);
        }
        if(fd>=0){
            close(fd);
        }
    }
    float consine_similarity(const vector<float>& emb1,const float* emb2,int dim=128) const{
            if(emb1.size()!=dim){cout<<"Number of Dimensions doesn't match"<<endl;return INT32_MIN;}
            int n_dimensions=dim;float dot_prod=0.0f;float magnitude1=0.0f;float magnitude2=0.0f;
            for(int i=0;i<n_dimensions;i++){
                dot_prod+=emb1[i]*emb2[i];
                magnitude1+=emb1[i]*emb1[i];
                magnitude2+=emb2[i]*emb2[i];
            }
            if(magnitude1==0.0f||magnitude2==0.0f){return 0.0f;}
            return dot_prod/(sqrt(magnitude1)*sqrt(magnitude2));
        }
         vector<int> search(vector<float>& query,int k=5) const {
            priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> Q;
            for(int i=0;i<num_vectors;i++){
                Q.push({consine_similarity(query,&(embeddings[i*n_dimensions]),n_dimensions),ids[i]});
                while(Q.size()>k){
                    Q.pop();
                }
            }
            vector<int> results;
            for(int i=0;i<k;i++){
                int id=Q.top().second;Q.pop();
                results.push_back(id);
            }
            std::reverse(results.begin(),results.end());
            return results;
        }
};

struct alignas(1024) IndexHeader{
    size_t num_nodes;
    int max_degree;
    int dimensions;
    int entry_point;
};

template<int n_DIMS, int max_degree>

struct alignas(1024) DiskNode{
    float vector_data[n_DIMS];//n_dim x 4 bytes
    int num_neighbours;//4 bytes
    int neighbours[max_degree];//max_degreex4 bytes
    // char padding[1024-(sizeof(float)*n_DIMS)-sizeof(int)-(sizeof(int)*max_degree)];
};

template<int n_DIMS,int max_degree>

class alignas(4096) VAMANA{
    private:
    int fd;
    size_t total_filesize;
    void* mapped_data;
    
    public:
    IndexHeader* header;
    DiskNode<n_DIMS,max_degree>* nodes;
    
    float l2_distance(const float* v1,const float*v2){
        // int n=header->dimensions;
        __m256 sum =_mm256_setzero_ps();
        for(int i=0;i<n_DIMS;i+=8){
            __m256 va=_mm256_loadu_ps(&v1[i]);
            __m256 vb=_mm256_loadu_ps(&v2[i]);
            __m256 diff= _mm256_sub_ps(va,vb);
            sum=_mm256_fmadd_ps(diff,diff,sum);
        }
        __m128 vlow=_mm256_castps256_ps128(sum);
        __m128 vhigh=_mm256_extractf128_ps(sum,1);
        vlow=_mm_add_ps(vlow,vhigh);
        __m128 _shuffle =_mm_movehdup_ps(vlow);
        __m128 sums= _mm_add_ps(vlow,_shuffle);
        _shuffle=_mm_movehl_ps(_shuffle,sums);
        sums=_mm_add_ss(sums,_shuffle);
        return _mm_cvtss_f32(sums);
    }

    pair<vector<pair<float,int>>,vector<int>> greedy_search(int start_node,const float* query,int k,int L,vector<uint8_t>& visited){
        //min_heap
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> min_heap;
        //max_heap
        priority_queue<pair<float,int>> top_L;
        vector<int> all_visited;
        float start_dist=l2_distance(nodes[start_node].vector_data,query);
        min_heap.push({start_dist,start_node});
        top_L.push({start_dist,start_node});
        visited[start_node]=1;
        all_visited.push_back(start_node);
        while(!min_heap.empty()){
            pair<float,int> curr=min_heap.top();min_heap.pop();
            if(top_L.size()==(size_t)L&&curr.first>top_L.top().first){break;}
            // if(top_L.size()<L||curr.first<top_L.top().first){
            //     top_L.push(curr);
            // }
            // visited[curr.second]=1;
            // all_visited.push_back(curr.second);
            for(int i=0;i<nodes[curr.second].num_neighbours;i++){
                int neighbour=nodes[curr.second].neighbours[i];
                if(visited[neighbour]==1){continue;}
                visited[neighbour]=1;
                all_visited.push_back(neighbour);
                if(i+1 < nodes[curr.second].num_neighbours){
                    __builtin_prefetch(&nodes[nodes[curr.second].neighbours[i+1]],0,3);
                }
                float dist=l2_distance(nodes[neighbour].vector_data,query);
                if(top_L.size()<L||dist<top_L.top().first){
                    min_heap.push({dist,neighbour});
                    top_L.push({dist,neighbour});
                    while(top_L.size()>L){top_L.pop();}
                }
            }
        }
        vector<pair<float,int>> results;
        while(!top_L.empty()){
            results.push_back(top_L.top());
            top_L.pop();
        }
        reverse(results.begin(),results.end());
        if((int)results.size()>k){
            results.resize(k);
        }
        for(int i=0;i<all_visited.size();i++){
            visited[all_visited[i]]=0;
        }
        return {results,all_visited};

    }
    void robust_prune(int p, vector<int>& V,float alpha, int R){
        thread_local vector<int>candidate_set;
        thread_local vector<pair<float,int>> candidates;
        thread_local vector<int> new_neighbours;

        candidate_set.clear();
        candidates.clear();
        new_neighbours.clear();
        candidate_set.reserve(V.size()+nodes[p].num_neighbours);
        for(int v:V){
            if(v!=p) candidate_set.push_back(v);
        }
        for(int i=0;i<nodes[p].num_neighbours;i++){
            int neighbour=nodes[p].neighbours[i];
            if(neighbour!=p) candidate_set.push_back(neighbour);
        }
        sort(candidate_set.begin(),candidate_set.end());
        candidate_set.erase(unique(candidate_set.begin(),candidate_set.end()),candidate_set.end());

        for(int v:candidate_set){
            candidates.push_back({l2_distance(nodes[v].vector_data,nodes[p].vector_data),v});
        }
        sort(candidates.begin(),candidates.end());
        new_neighbours.reserve(R);
        for(auto& curr:candidates){
            bool keep=true;
            for( int v: new_neighbours){
                float v_dist=l2_distance(nodes[v].vector_data,nodes[curr.second].vector_data);
                if((alpha*alpha*v_dist)<=curr.first){
                    keep=false;
                    break;
                }
            }
            if(keep){
                new_neighbours.push_back(curr.second);
                if(new_neighbours.size()==R){break;}
            }
        }
        nodes[p].num_neighbours=new_neighbours.size();
        for(int i=0;i<new_neighbours.size();i++){
            nodes[p].neighbours[i]=new_neighbours[i];
        }
    }
    struct alignas(64) Paddedlock{
        omp_lock_t lock;
    };
    void build_index(float alpha){
        size_t n = header->num_nodes;
        int L = 150;

        Paddedlock* node_locks = new Paddedlock[n];
        for(size_t i = 0; i < n; i++){
            omp_init_lock(&node_locks[i].lock);
        }

        for(float curr_alpha : {1.0f, alpha}){
            cout << "Building Vamana with Alpha: " << curr_alpha << endl;
            #pragma omp parallel
            {
                vector<uint8_t> local_visited(n, 0);
                #pragma omp for schedule(dynamic, 128)
                for(size_t i = 0; i < n; i++){
                    auto curr = greedy_search(header->entry_point, nodes[i].vector_data, 1, L, local_visited);

                    robust_prune(i, curr.second, curr_alpha, header->max_degree);
                    
                    for(int j = 0; j < nodes[i].num_neighbours; j++){
                        int neighbour = nodes[i].neighbours[j];
                        omp_set_lock(&node_locks[neighbour].lock);
                        
                        bool already_neighbour = false;
                        for(int k = 0; k < nodes[neighbour].num_neighbours; k++){
                            if(nodes[neighbour].neighbours[k] == (int)i) { already_neighbour = true; break; }
                        }
                        
                        if(!already_neighbour){
                            if(nodes[neighbour].num_neighbours < header->max_degree){
                                nodes[neighbour].neighbours[nodes[neighbour].num_neighbours++] = i;
                            } else {
                                vector<int> neighbour_candidates;
                                neighbour_candidates.push_back(i);
                                robust_prune(neighbour, neighbour_candidates, curr_alpha, header->max_degree);
                            }
                        }
                        omp_unset_lock(&node_locks[neighbour].lock);
                    }
                }
            }
        }
        for(size_t i = 0; i < n; i++){
            omp_destroy_lock(&node_locks[i].lock);
        }
        delete[] node_locks;
        
        cout << "Vamana Index Build Completed" << endl;
    }
    
    VAMANA(string filename, size_t n){
        fd=open(filename.c_str(),O_RDWR|O_CREAT|O_TRUNC,0644);
        if(fd<0){
            cerr<< "failed to open "<<filename<<endl;
            exit(1);
        }
        total_filesize=sizeof(IndexHeader)+(n*sizeof(DiskNode<n_DIMS,max_degree>));
        
        if(ftruncate(fd,total_filesize)<0){
            cerr<< "failed to open "<<filename<<endl;
            exit(1);   
            }
            mapped_data=mmap(nullptr,total_filesize,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            if(mapped_data==MAP_FAILED){cerr<<"mmap failed"<<endl;exit(1);}
            madvise(mapped_data,total_filesize,MADV_RANDOM);
            header=static_cast<IndexHeader*>(mapped_data);
            nodes=reinterpret_cast<DiskNode<n_DIMS,max_degree>*>(static_cast<char*>(mapped_data)+sizeof(IndexHeader));
            header->num_nodes=n;
            header->max_degree=max_degree;
            header->dimensions=n_DIMS;
            //temporary entry will be updated in coming function
            header->entry_point=0; 

            cout<<"Mapped Data"<< total_filesize/(1024*1024)<<" MB saved in Disk"<<endl;
        }
        void initialize_random_graph(){
            cout<<"Generating a random Vamana graph"<<endl;
            size_t n=header->num_nodes;
            #pragma omp parallel
            {
                mt19937 rng(42+omp_get_thread_num());
                uniform_int_distribution<int> dist(0,n-1);
                #pragma omp for
                for(size_t i=0;i<n;i++){
                    nodes[i].num_neighbours=max_degree;
                    for(int j=0;j<max_degree;j++){
                        int neighbour=dist(rng);
                        while(neighbour==i){neighbour=dist(rng);}
                        nodes[i].neighbours[j]=neighbour;
                    }
                }
            }
            cout<<"Done Generating random Vamana Graph"<<endl;
        }
        void load_data_and_get_medoid(float* flat_vectors){
            cout<<"Loading Data from flat vectors"<<endl;
            size_t n=header->num_nodes;
            vector<double> global_sum(n_DIMS,0.0);
            #pragma omp parallel
            {
                vector<double> local_sum(n_DIMS,0.0);
                #pragma omp for
                for(size_t i=0;i<n;i++){
                    for(int j=0;j<n_DIMS;j++){
                        float val=flat_vectors[i*n_DIMS+j];
                        local_sum[j]+=val;
                        nodes[i].vector_data[j]=val;
                    }
                    nodes[i].num_neighbours=0;
                }
                #pragma omp critical
                {
                    for(int d=0;d<n_DIMS;d++){
                        global_sum[d]+=local_sum[d];
                    }
                }
            }
            vector<float> centroid(n_DIMS,0.0);
            for(int d=0;d<n_DIMS;d++){centroid[d]=global_sum[d]/n;}
            cout<<"finding Medoid"<<endl;
            double min_dist=INT32_MAX;
            int medoid_id=-1;
            #pragma omp parallel
            {
                double local_min_dist=INT32_MAX;
                int local_medoid_id=-1;
                #pragma omp for
                for(size_t i=0;i<n;i++){
                    double dist=0.0;
                    for(int d=0;d<n_DIMS;d++){
                        float temp=nodes[i].vector_data[d]-centroid[d];
                        dist+=temp*temp;
                    }
                    if(dist<local_min_dist){
                        local_min_dist=dist;
                        local_medoid_id=i;
                    }
                }
                #pragma omp critical
                {
                    if(local_min_dist<min_dist){
                        min_dist=local_min_dist;
                        medoid_id=local_medoid_id;
                    }
                }
            }
            header->entry_point=medoid_id;
            cout<<"Found Medoid Id: "<<medoid_id<<endl;
        }
        void sync_to_disk(){
            cout<<"Flushing Data to disk"<<endl;
            msync(mapped_data,total_filesize,MS_SYNC);
            cout<<"done Flush is complete"<<endl;
        }

        ~VAMANA(){
            sync_to_disk();
            munmap(mapped_data,total_filesize);
            close(fd);
        }

};


//===========================================================================
// FILTERED DISKANN
//===========================================================================

const int MAX_LABELS=64;
struct alignas(1024) Filtered_IndexHeader{
    size_t num_nodes;
    int filter_medoids[MAX_LABELS];
    int max_degree;
    int dimensions;
    int num_labels;
};

template<int n_DIMS, int max_degree>

struct alignas(1024) Filtered_DiskNode{
    float vector_data[n_DIMS];//n_dim x 4 bytes
    int num_neighbours;//4 bytes
    int neighbours[max_degree];//max_degreex4 bytes
    int dummy_padding;
    uint64_t bitmap;
    // char padding[1024-(sizeof(float)*n_DIMS)-2*sizeof(int)-(sizeof(int)*max_degree)-sizeof(uint64_t)];
};

template<int n_DIMS,int max_degree>
class alignas(4096) Filtered_VAMANA{
    private:
    int fd;
    size_t total_filesize;
    void* mapped_data;
    
    public:
    Filtered_IndexHeader* header;
    vector<int> global_label_counts;
    Filtered_DiskNode<n_DIMS,max_degree>* nodes;
    
    float l2_distance(const float* v1,const float*v2){
        // int n=header->dimensions;
        __m256 sum =_mm256_setzero_ps();
        for(int i=0;i<n_DIMS;i+=8){
            __m256 va=_mm256_loadu_ps(&v1[i]);
            __m256 vb=_mm256_loadu_ps(&v2[i]);
            __m256 diff= _mm256_sub_ps(va,vb);
            sum=_mm256_fmadd_ps(diff,diff,sum);
        }
        __m128 vlow=_mm256_castps256_ps128(sum);
        __m128 vhigh=_mm256_extractf128_ps(sum,1);
        vlow=_mm_add_ps(vlow,vhigh);
        __m128 _shuffle =_mm_movehdup_ps(vlow);
        __m128 sums= _mm_add_ps(vlow,_shuffle);
        _shuffle=_mm_movehl_ps(_shuffle,sums);
        sums=_mm_add_ss(sums,_shuffle);
        return _mm_cvtss_f32(sums);
    }

    pair<vector<pair<float,int>>,vector<int>> Filtered_greedy_search(int start_nodes[MAX_LABELS],const float* query,uint64_t query_bitmap,int k,int L,vector<uint8_t>& visited){
        //min_heap
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> min_heap;
        //max_heap
        priority_queue<pair<float,int>> top_L;
        vector<int> all_visited;
        // uint64_t temp=1ULL;
        uint64_t mask=query_bitmap;
        while(mask!=0){
            int label_id=__builtin_ctzll(mask);
            mask&=(mask-1);
                int start_node=start_nodes[label_id];
                if(start_node==-1){continue;}
                if(visited[start_node]!=1){
                    float start_dist=l2_distance(nodes[start_node].vector_data,query);
                    min_heap.push({start_dist,start_node});
                    top_L.push({start_dist,start_node});
                    while(top_L.size()>(size_t)L){top_L.pop();}
                    visited[start_node]=1;
                    all_visited.push_back(start_node);
                }
        }
        if(min_heap.empty()){
            //no vectors with that filter
            cerr<<"No vectors with filter resort to Standard Vamana"<<endl;exit(1);
        }
        while(!min_heap.empty()){
            pair<float,int> curr=min_heap.top();min_heap.pop();
            if(top_L.size()==(size_t)L&&curr.first>top_L.top().first){break;}
            for(int i=0;i<nodes[curr.second].num_neighbours;i++){
                int neighbour=nodes[curr.second].neighbours[i];
                if(visited[neighbour]==1){continue;}
                if((nodes[neighbour].bitmap&query_bitmap)==0){continue;}
                visited[neighbour]=1;
                all_visited.push_back(neighbour);
                if(i+1 < nodes[curr.second].num_neighbours){
                    __builtin_prefetch(&nodes[nodes[curr.second].neighbours[i+1]],0,3);
                }
                float dist=l2_distance(nodes[neighbour].vector_data,query);
                if(top_L.size()<L||dist<top_L.top().first){
                    min_heap.push({dist,neighbour});
                    top_L.push({dist,neighbour});
                    while(top_L.size()>L){top_L.pop();}
                }
            }
        }
        vector<pair<float,int>> results;
        while(!top_L.empty()){
            results.push_back(top_L.top());
            top_L.pop();
        }
        reverse(results.begin(),results.end());
        if((int)results.size()>k){
            results.resize(k);
        }
        for(int i=0;i<all_visited.size();i++){
            visited[all_visited[i]]=0;
        }

        return {results,all_visited};

    }
    void Filtered_robust_prune(int p, vector<int>& V, float alpha, int R) {
    thread_local vector<int> candidate_set;
    thread_local vector<pair<float,int>> candidates;
    
    candidate_set.clear();
    candidates.clear();
    candidate_set.reserve(V.size() + nodes[p].num_neighbours);
    uint64_t p_bitmap = nodes[p].bitmap;
    
    for(int v : V) { if(v != p) candidate_set.push_back(v); }
    for(int i = 0; i < nodes[p].num_neighbours; i++){
        int neighbour = nodes[p].neighbours[i];
        if(neighbour != p) candidate_set.push_back(neighbour);
    }

    sort(candidate_set.begin(), candidate_set.end());
    candidate_set.erase(unique(candidate_set.begin(), candidate_set.end()), candidate_set.end());

    for(int v : candidate_set){
        candidates.push_back({l2_distance(nodes[v].vector_data, nodes[p].vector_data), v});
    }
    sort(candidates.begin(), candidates.end());

    int label_thresholds[MAX_LABELS] = {0};
    int current_label_counts[MAX_LABELS] = {0};
    uint64_t temp_p = p_bitmap;
    int total_freq = 0;
    
    while(temp_p != 0){
        int l = __builtin_ctzll(temp_p);
        total_freq += global_label_counts[l];
        temp_p &= (temp_p - 1);
    }
    
    //Allocating min_thresholds proportionally
    temp_p = p_bitmap;
    while(temp_p != 0){
        int l = __builtin_ctzll(temp_p);
        if (total_freq > 0) {
            // Proportional allocation
            label_thresholds[l] = std::max(8, (int)(R * ((float)global_label_counts[l] / total_freq)));
        }
        temp_p &= (temp_p - 1);
    }

    vector<int> new_neighbours;
    vector<int> overflow_candidates;
    new_neighbours.reserve(R);

    for(auto& curr : candidates) {
        if (new_neighbours.size() == (size_t)R) break; 

        uint64_t I = (nodes[curr.second].bitmap & p_bitmap);
        if (I == 0) continue;

        bool keep = true;
        for(int v : new_neighbours) {
            if ((nodes[v].bitmap & I) != I) continue;
            float v_dist = l2_distance(nodes[v].vector_data, nodes[curr.second].vector_data);
            if (v_dist <= alpha * alpha * curr.first) {
                keep = false; break;
            }
        }

        if (keep) {
            bool under_quota = false;
            uint64_t temp_I = I;            
            while(temp_I != 0) {
                int l = __builtin_ctzll(temp_I);
                if (current_label_counts[l] < label_thresholds[l]) { 
                    under_quota = true; break; 
                }
                temp_I &= (temp_I - 1);
            }
            if (under_quota) {
                new_neighbours.push_back(curr.second);
                temp_I = I;
                while(temp_I != 0) {
                    int l = __builtin_ctzll(temp_I);
                    current_label_counts[l]++;
                    temp_I &= (temp_I - 1);
                }
            } else {
                overflow_candidates.push_back(curr.second);
            }
        }
    }
    for (int v : overflow_candidates) {
        if (new_neighbours.size() == (size_t)R) break;
        new_neighbours.push_back(v);
    }

    for(int i = 0; i < new_neighbours.size(); i++){
        nodes[p].neighbours[i] = new_neighbours[i];
    }
    nodes[p].num_neighbours = new_neighbours.size();
}
    struct alignas(64) Paddedlock{
        omp_lock_t lock;
    };
    void Filtered_build_index(float alpha){
        size_t n = header->num_nodes;
        int L = 200;
        cout<<"Check"<<endl;
        Paddedlock* node_locks = new Paddedlock[n];
        for(size_t i = 0; i < n; i++){
            omp_init_lock(&node_locks[i].lock);
        }
        vector<int> insertion_order(n);
        for(size_t i=0;i<n;i++){insertion_order[i]=i;}
        mt19937 rng(42);
        shuffle(insertion_order.begin(),insertion_order.end(),rng);
        for(float curr_alpha : {1.0f, alpha}){
            cout << "Building Vamana with Alpha: " << curr_alpha << endl;
            #pragma omp parallel
            {
                vector<uint8_t> local_visited(n, 0);
                #pragma omp for schedule(dynamic, 128)
                for(size_t iter = 0; iter < n; iter++){
                    int i = insertion_order[iter];

                    vector<int> all_candidates;
                    uint64_t mask = nodes[i].bitmap;

                    while(mask != 0) {
                        int label_id = __builtin_ctzll(mask);
                        mask &= (mask - 1);
                        
                        uint64_t single_label_mask = (1ULL << label_id);
                        auto curr = Filtered_greedy_search(header->filter_medoids, nodes[i].vector_data, single_label_mask, L, L, local_visited);                        
                        for(int v : curr.second) {
                            all_candidates.push_back(v);
                        }
                    }

                    sort(all_candidates.begin(), all_candidates.end());
                    all_candidates.erase(unique(all_candidates.begin(), all_candidates.end()), all_candidates.end());

                    omp_set_lock(&node_locks[i].lock);
                    Filtered_robust_prune(i, all_candidates, curr_alpha, header->max_degree);
                    vector<int> current_neighbours(nodes[i].num_neighbours);
                    for(int j = 0; j < nodes[i].num_neighbours; j++){
                        current_neighbours[j] = nodes[i].neighbours[j];
                    }
                    omp_unset_lock(&node_locks[i].lock);

                    for(int j = 0; j < current_neighbours.size(); j++){
                        int neighbour = current_neighbours[j];
                        omp_set_lock(&node_locks[neighbour].lock);
                        
                        bool already_neighbour = false;
                        for(int k = 0; k < nodes[neighbour].num_neighbours; k++){
                            if(nodes[neighbour].neighbours[k] == (int)i) { already_neighbour = true; break; }
                        }
                        
                        if(!already_neighbour){
                            if(nodes[neighbour].num_neighbours < header->max_degree){
                                int count = nodes[neighbour].num_neighbours;
                                nodes[neighbour].neighbours[count] = i;
                                nodes[neighbour].num_neighbours = count + 1; 
                            } else {
                                vector<int> neighbour_candidates;
                                neighbour_candidates.push_back(i);
                                Filtered_robust_prune(neighbour, neighbour_candidates, curr_alpha, header->max_degree);
                            }
                        }
                        omp_unset_lock(&node_locks[neighbour].lock);
                    }
                }
            }
        }
        for(size_t i = 0; i < n; i++){
            omp_destroy_lock(&node_locks[i].lock);
        }
        delete[] node_locks;
        
        cout << "Vamana Index Build Completed" << endl;
    }
    void initialize_filtered_random_graph() {
    cout << "Initializing Label-Aware Random Graph (Exact Sampling + Medoid Routing)..." << endl;
    size_t n = header->num_nodes;
    vector<vector<int>> label_lists(MAX_LABELS);
    for(size_t i = 0; i < n; i++) {
        uint64_t mask = nodes[i].bitmap;
        while(mask != 0) {
            int label_id = __builtin_ctzll(mask);
            label_lists[label_id].push_back(i);
            mask &= (mask - 1);
        }
    }

    #pragma omp parallel
    {
        mt19937 rng(42 ^ omp_get_thread_num()); 

        #pragma omp for schedule(dynamic, 128)
        for (size_t i = 0; i < n; i++) {
            nodes[i].num_neighbours = 0;    
            uint64_t mask = nodes[i].bitmap;
            int num_labels = __builtin_popcountll(mask);
            if (num_labels == 0) continue;            
            int edges_per_label = std::max(2, 15 / num_labels); 
            
            uint64_t temp_mask = mask;
            while(temp_mask != 0) {
                int label_id = __builtin_ctzll(temp_mask);
                temp_mask &= (temp_mask - 1);
                
                const auto& list = label_lists[label_id];
                int list_size = list.size();
                
                if (list_size <= 1) continue; 
                
                uniform_int_distribution<int> dist(0, list_size - 1);
                int added_for_this_label = 0;
                int attempts = 0;                
                while(added_for_this_label < edges_per_label && attempts < 50 && nodes[i].num_neighbours < header->max_degree) {
                    int rand_node = list[dist(rng)]; 
                    if (rand_node != (int)i) {
                        bool exists = false;
                        for(int k = 0; k < nodes[i].num_neighbours; k++){
                            if(nodes[i].neighbours[k] == rand_node) { exists = true; break; }
                        }
                        if(!exists) {
                            nodes[i].neighbours[nodes[i].num_neighbours++] = rand_node;
                            added_for_this_label++;
                        }
                    }
                    attempts++;
                }

                int medoid = header->filter_medoids[label_id];
                if (medoid != -1 && medoid != (int)i && nodes[i].num_neighbours < header->max_degree) {
                    bool exists = false;
                    for(int k = 0; k < nodes[i].num_neighbours; k++){
                        if(nodes[i].neighbours[k] == medoid) { exists = true; break; }
                    }
                    if(!exists) {
                        nodes[i].neighbours[nodes[i].num_neighbours++] = medoid;
                    }
                }
            }
        }
    }
}
    Filtered_VAMANA(string filename, size_t n){
        fd=open(filename.c_str(),O_RDWR|O_CREAT|O_TRUNC,0644);
        if(fd<0){
            cerr<< "failed to open "<<filename<<endl;
            exit(1);
        }
        total_filesize=sizeof(Filtered_IndexHeader)+(n*sizeof(Filtered_DiskNode<n_DIMS,max_degree>));
        
        if(ftruncate(fd,total_filesize)<0){
            cerr<< "failed to open "<<filename<<endl;
            exit(1);   
            }
            mapped_data=mmap(nullptr,total_filesize,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
            if(mapped_data==MAP_FAILED){cerr<<"mmap failed"<<endl;exit(1);}
            madvise(mapped_data,total_filesize,MADV_RANDOM);
            header=static_cast<Filtered_IndexHeader*>(mapped_data);
            nodes=reinterpret_cast<Filtered_DiskNode<n_DIMS,max_degree>*>(static_cast<char*>(mapped_data)+sizeof(Filtered_IndexHeader));
            header->num_nodes=n;
            header->max_degree=max_degree;
            header->dimensions=n_DIMS;
            //temporary entry will be updated in coming function
            // header->entry_point=0;

            cout<<"Mapped Data"<< total_filesize/(1024*1024)<<" MB saved in Disk"<<endl;
        }
        void load_data_and_get_medoids(const float* flat_vectors,const uint64_t* labels){
            cout<<"Loading Data from flat vectors"<<endl;
            size_t n=header->num_nodes;
            vector<vector<double>> global_sums(MAX_LABELS,vector<double>(n_DIMS,0.0));
            vector<int> global_counts(MAX_LABELS,0);
            #pragma omp parallel
            {
            vector<vector<double>> local_sums(MAX_LABELS,vector<double>(n_DIMS,0.0));
            vector<int> local_counts(MAX_LABELS,0);
                #pragma omp for
                for(size_t i=0;i<n;i++){
                    uint64_t mask=labels[i];
                    nodes[i].bitmap=mask;
                    nodes[i].num_neighbours=0;
                    float curr_vector[n_DIMS];
                    for(int j=0;j<n_DIMS;j++){
                        float val=flat_vectors[i*n_DIMS+j];
                        curr_vector[j]=val;
                        nodes[i].vector_data[j]=val;
                    }
                    while(mask!=0){
                        int label_id=__builtin_ctzll(mask);
                        for(int d=0;d<n_DIMS;d++){
                            local_sums[label_id][d]+=curr_vector[d];
                        }
                        local_counts[label_id]++;
                        mask&=(mask-1);
                    }
                }
                #pragma omp critical
                {
                    for(int i=0;i< MAX_LABELS;i++){
                        global_counts[i]+=local_counts[i];
                        for(int d=0;d<n_DIMS;d++){
                            global_sums[i][d]+=local_sums[i][d];
                        }
                        
                    }
                }
            }
            vector<vector<float>> centroids(MAX_LABELS,vector<float>(n_DIMS,0.0));
            for(int i=0;i<MAX_LABELS;i++){
                if(global_counts[i]>0){

                    for(int d=0;d<n_DIMS;d++){centroids[i][d]=(float)global_sums[i][d]/global_counts[i];}
                }
            }
            cout<<"finding Top-K candidates for Randomized Load balancing"<<endl;
            global_label_counts=global_counts;
            int K=10;
            vector<priority_queue<pair<float,int>>> global_top_k(MAX_LABELS);
            #pragma omp parallel
            {
                vector<priority_queue<pair<float,int>>> local_top_k(MAX_LABELS);
                #pragma omp for schedule(dynamic,128)
                for(size_t i=0;i<n;i++){
                    uint64_t mask=nodes[i].bitmap;
                    while(mask!=0){
                        int label_id=__builtin_ctzll(mask);
                        mask&=(mask-1);
                        double dist=l2_distance(nodes[i].vector_data,centroids[label_id].data());
                        local_top_k[label_id].push({dist,(int)i});
                        if(local_top_k[label_id].size()>(size_t)K){
                            local_top_k[label_id].pop();
                        }
                    }
                }
                #pragma omp critical
                {
                    for(int i=0;i< MAX_LABELS;i++){
                        while(!local_top_k[i].empty()){
                            global_top_k[i].push(local_top_k[i].top());
                            local_top_k[i].pop();
                            if(global_top_k[i].size()>(size_t)K){
                                global_top_k[i].pop();
                            }
                        }

                    }
                }
            }
            vector<int> medoid_usage_count(n,0);
            for(int i=0;i<MAX_LABELS;i++){
                if(global_counts[i]==0||global_top_k[i].empty()){
                    header->filter_medoids[i]=-1;continue;
                }

                int min_usage=INT32_MAX;
                int best_candidate=-1;;
                while(!global_top_k[i].empty()){
                    int candidate_id=global_top_k[i].top().second;
                    global_top_k[i].pop();

                    int usage=medoid_usage_count[candidate_id];
                    if(usage <= min_usage){
                        min_usage=usage;
                        best_candidate=candidate_id;
                    }
                }
                if(best_candidate!=-1){
                    header->filter_medoids[i]=best_candidate;
                medoid_usage_count[best_candidate]++;
                }
                else{
                    header->filter_medoids[i]=-1;
                    // medoid_usage_count[0]++;
                }
            }
            cout<<"Found ALL Medoid Ids "<<endl;
        }
        void sync_to_disk(){
            cout<<"Flushing Data to disk"<<endl;
            msync(mapped_data,total_filesize,MS_SYNC);
            cout<<"done Flush is complete"<<endl;
        }

        ~Filtered_VAMANA(){
            sync_to_disk();
            munmap(mapped_data,total_filesize);
            close(fd);
        }

};