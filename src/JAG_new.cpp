#pragma once

#include <iostream>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stack>
#include <cctype>
#include <iomanip>
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
#include "retriever.hpp"
#include <omp.h>


struct alignas(64) Nitro_IndexHeader {
    size_t num_nodes;
    int max_degree;
    int dimensions;
    int global_medoid_id; 
    float global_sigma_subset;
    float global_sigma_range;
};

template<int n_DIMS, int max_degree>
struct alignas(1024) Nitro_DiskNode {
    float vector_data[n_DIMS];        
    int num_neighbours;               
    int neighbours[max_degree];   
    int dummy_padding;    
    uint64_t bitmap;     
    float numeric_attr;   
};

struct NitroCandidate {
    float penalty; 
    float v_dist;   
    int id;         

    bool operator<(const NitroCandidate& other) const {
        if (std::abs(penalty - other.penalty) > 1e-5f) return penalty < other.penalty;
        return v_dist < other.v_dist;
    }

    bool operator>(const NitroCandidate& other) const {
        if (std::abs(penalty - other.penalty) > 1e-5f) return penalty > other.penalty;
        return v_dist > other.v_dist;
    }
};

struct alignas(64) PaddedLock {
    omp_lock_t lock;
};

template<int n_DIMS, int max_degree>
class NitroRAG_Vamana {
private:
    int fd;
    size_t total_filesize;
    void* mapped_data;

public:
    Nitro_IndexHeader* header;
    Nitro_DiskNode<n_DIMS, max_degree>* nodes;

    inline float l2_distance(const float* v1, const float* v2) const {
        __m256 sum = _mm256_setzero_ps();
        for(int i = 0; i < n_DIMS; i += 8) {
            __m256 va = _mm256_loadu_ps(&v1[i]);
            __m256 vb = _mm256_loadu_ps(&v2[i]);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        __m128 vlow = _mm256_castps256_ps128(sum);
        __m128 vhigh = _mm256_extractf128_ps(sum, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 _shuffle = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, _shuffle);
        _shuffle = _mm_movehl_ps(_shuffle, sums);
        sums = _mm_add_ss(sums, _shuffle);
        return _mm_cvtss_f32(sums);
    }


    inline float get_query_penalty(uint64_t n_bit, float n_attr, uint64_t q_bit, float q_min, float q_max) const {
        int missing_bits = (q_bit == 0) ? 0 : __builtin_popcountll(q_bit & ~n_bit);
        float norm_sub = (float)(missing_bits/header->global_sigma_subset);
        float range_dist = 0.0f;
        if (n_attr < q_min) range_dist =q_min -n_attr;
        else if (n_attr > q_max) range_dist =n_attr-q_max;
        float norm_rng = range_dist/header->global_sigma_range;
        return norm_sub +norm_rng;
    }

    inline float get_node_penalty(uint64_t bit_p, float attr_p, uint64_t bit_q, float attr_q) const {
        int diff_bits = __builtin_popcountll(bit_p ^ bit_q);
        float norm_sub = (float)diff_bits / header->global_sigma_subset;
        
        float range_dist = std::abs(attr_p - attr_q);
        float norm_rng = range_dist / header->global_sigma_range;

        return norm_sub + norm_rng;
    }

    void compute_global_statistics() {
        size_t n = header->num_nodes;
        const int BATCH_SIZE = 10000;
        int MAX_BATCHES = 50;

        double total_sum_bits = 0.0, total_sq_bits = 0.0;
        double total_sum_rng = 0.0,  total_sq_rng = 0.0;
        int total_samples = 0;
        double prev_sigma_sub = -1.0, prev_sigma_rng = -1.0;
        mt19937 rng(42);
        uniform_int_distribution<int> dist_nodes(0, n - 1);

        for (int b = 0; b < MAX_BATCHES; b++) {
            double batch_sum_bits = 0.0, batch_sq_bits = 0.0;
            double batch_sum_rng = 0.0,  batch_sq_rng = 0.0;
            #pragma omp parallel
            {
                mt19937 local_rng(42 ^ omp_get_thread_num() ^ b);                 
                #pragma omp for reduction(+:batch_sum_bits, batch_sq_bits, batch_sum_rng, batch_sq_rng)
                for(int i = 0; i < BATCH_SIZE; i++) {
                    int id1 = dist_nodes(local_rng);
                    int id2 = dist_nodes(local_rng);
                    double h_dist = (double)__builtin_popcountll(nodes[id1].bitmap ^ nodes[id2].bitmap);
                    double r_dist = std::abs(nodes[id1].numeric_attr - nodes[id2].numeric_attr);
                    batch_sum_bits += h_dist;
                    batch_sq_bits  += (h_dist * h_dist);
                    batch_sum_rng += r_dist;
                    batch_sq_rng  += (r_dist * r_dist);
                }
            }

            total_sum_bits+= batch_sum_bits;
            total_sq_bits+= batch_sq_bits;
            total_sum_rng+= batch_sum_rng;
            total_sq_rng+= batch_sq_rng;
            total_samples+= BATCH_SIZE;

            double mean_bits= total_sum_bits/total_samples;
            double mean_rng = total_sum_rng/total_samples;

            double var_bits= (total_sq_bits / total_samples) - (mean_bits * mean_bits);
            double var_rng= (total_sq_rng / total_samples) - (mean_rng * mean_rng);

            if (var_bits<0) var_bits= 0;
            if (var_rng<0) var_rng= 0;

            double current_sigma_sub = sqrt(var_bits);
            double current_sigma_rng = sqrt(var_rng);

            if (b > 0) { 
                double diff_sub = std::abs(current_sigma_sub-prev_sigma_sub) /(prev_sigma_sub + 1e-6);
                double diff_rng = std::abs(current_sigma_rng-prev_sigma_rng) /(prev_sigma_rng + 1e-6);

                if (diff_sub < 0.005 && diff_rng < 0.005) {
                    header->global_sigma_subset = current_sigma_sub;
                    header->global_sigma_range  = current_sigma_rng;
                    cout << "Distribution Converged after " << total_samples << " samples" << endl;
                    break;
                }
            }
            prev_sigma_sub = current_sigma_sub;
            prev_sigma_rng = current_sigma_rng;
            header->global_sigma_subset = current_sigma_sub;
            header->global_sigma_range  = current_sigma_rng;
        }
        if (header->global_sigma_subset < 1e-6f) header->global_sigma_subset = 1.0f;
        if (header->global_sigma_range < 1e-6f) header->global_sigma_range = 1.0f;
        printf("Sigma Subset: %.4f and Sigma Range: %.4f\n",header->global_sigma_subset, header->global_sigma_range);
    }

    pair<vector<NitroCandidate>, vector<int>> greedy_search(const float* q_vec, uint64_t q_bit, float q_min, float q_max, int L, vector<uint8_t>& visited) {
        priority_queue<NitroCandidate, vector<NitroCandidate>, greater<NitroCandidate>> min_heap;
        priority_queue<NitroCandidate> top_L;
        vector<int> all_visited;
        int start_node = header->global_medoid_id;
        float start_vdist = l2_distance(nodes[start_node].vector_data, q_vec);
        float start_pen = get_query_penalty(nodes[start_node].bitmap, nodes[start_node].numeric_attr, q_bit, q_min, q_max);
        NitroCandidate start_cand = {start_pen, start_vdist, start_node};
        
        min_heap.push(start_cand);
        top_L.push(start_cand);
        visited[start_node] = 1;
        all_visited.push_back(start_node);
        
        while(!min_heap.empty()){
            NitroCandidate curr = min_heap.top(); 
            min_heap.pop();
            if(top_L.size()==(size_t)L && curr > top_L.top()) { break;}
            for(int i = 0; i < nodes[curr.id].num_neighbours; i++){
                int neighbor = nodes[curr.id].neighbours[i];
                if(visited[neighbor] == 1) continue;
                visited[neighbor] = 1;
                all_visited.push_back(neighbor);
                if(i + 1 < nodes[curr.id].num_neighbours){
                    __builtin_prefetch(&nodes[nodes[curr.id].neighbours[i+1]], 0, 3);
                }
                float v_dist = l2_distance(nodes[neighbor].vector_data, q_vec);
                float pen = get_query_penalty(nodes[neighbor].bitmap, nodes[neighbor].numeric_attr, q_bit, q_min, q_max);
                NitroCandidate n_cand = {pen, v_dist, neighbor};
                if(top_L.size() < (size_t)L || top_L.top() > n_cand){
                    min_heap.push(n_cand);
                    top_L.push(n_cand);
                    if(top_L.size() > (size_t)L) { top_L.pop(); }
                }
            }
        }
        
        vector<NitroCandidate> results;
        while(!top_L.empty()){
            results.push_back(top_L.top());
            top_L.pop();
        }
        reverse(results.begin(), results.end()); 
        for(int v : all_visited) visited[v] = 0;
        
        return {results, all_visited};
    }

    void robust_prune(int p, vector<int>& V, float alpha, int R) {
        int budget_SEMANTIC = (R * 5) / 8;  
        int budget_VECTOR = R - budget_SEMANTIC; 
        
        vector<int> candidate_set;
        candidate_set.reserve(V.size() + nodes[p].num_neighbours);
        for(int v : V) { if(v != p) candidate_set.push_back(v); }
        for(int i = 0; i < nodes[p].num_neighbours; i++){
            if(nodes[p].neighbours[i] != p) candidate_set.push_back(nodes[p].neighbours[i]);
        }
        sort(candidate_set.begin(), candidate_set.end());
        candidate_set.erase(unique(candidate_set.begin(), candidate_set.end()), candidate_set.end());

        vector<int> final_neighbours;
        final_neighbours.reserve(R);
        vector<NitroCandidate> semantic_cands;
        for(int v : candidate_set){
            float p_v_pen = get_node_penalty(nodes[p].bitmap, nodes[p].numeric_attr, nodes[v].bitmap, nodes[v].numeric_attr);
            float p_v_dist = l2_distance(nodes[p].vector_data, nodes[v].vector_data);
            semantic_cands.push_back({p_v_pen, p_v_dist, v});
        }
        sort(semantic_cands.begin(), semantic_cands.end());

        vector<int> semantic_edges;
        for(auto& curr : semantic_cands) {
            if(semantic_edges.size() == (size_t)budget_SEMANTIC) break;

            bool keep = true;
            for(int v : semantic_edges) {
                float v_curr_pen = get_node_penalty(nodes[v].bitmap, nodes[v].numeric_attr, nodes[curr.id].bitmap, nodes[curr.id].numeric_attr);
                float v_curr_dist = l2_distance(nodes[v].vector_data, nodes[curr.id].vector_data);
                
                if (v_curr_pen <= curr.penalty && v_curr_dist <= alpha * alpha * curr.v_dist) {
                    keep = false; break;
                }
            }
            if(keep) {
                semantic_edges.push_back(curr.id);
                final_neighbours.push_back(curr.id);
            }
        }

        vector<NitroCandidate> vector_cands;
        for(int v : candidate_set){
            float p_v_dist = l2_distance(nodes[p].vector_data, nodes[v].vector_data);
            vector_cands.push_back({0.0f, p_v_dist, v}); 
        }
        sort(vector_cands.begin(), vector_cands.end(), [](const NitroCandidate& a, const NitroCandidate& b) {
            return a.v_dist < b.v_dist;
        });

        vector<int> vector_edges;
        for(auto& curr : vector_cands) {
            if(vector_edges.size() == (size_t)budget_VECTOR) break;            
            if(find(final_neighbours.begin(), final_neighbours.end(), curr.id) != final_neighbours.end()) {
                continue;
            }

            bool keep = true;
            for(int v : vector_edges) {
                float v_curr_dist = l2_distance(nodes[v].vector_data, nodes[curr.id].vector_data);
                if (v_curr_dist <= alpha * alpha * curr.v_dist) {
                    keep = false; break;
                }
            }
            if(keep) {
                vector_edges.push_back(curr.id);
                final_neighbours.push_back(curr.id);
            }
        }
        for(int i = 0; i < final_neighbours.size() && i < R; i++){
            nodes[p].neighbours[i] = final_neighbours[i];
        }
        nodes[p].num_neighbours = min((int)final_neighbours.size(), R);
    }

    void build_index(float alpha) {
        size_t n = header->num_nodes;
        int L_build = 150; 

        PaddedLock* node_locks = new PaddedLock[n];
        for(size_t i = 0; i < n; i++) omp_init_lock(&node_locks[i].lock);

        vector<int> insertion_order(n);
        for(size_t i=0; i<n; i++) insertion_order[i] = i;
        mt19937 rng(42);
        shuffle(insertion_order.begin(), insertion_order.end(), rng);
        for(float curr_alpha : {1.0f, alpha}) {
            cout << "[NitroRAG] Building multiplexed graph pass. Alpha: " << curr_alpha << endl;
            #pragma omp parallel
            {
                vector<uint8_t> local_visited(n, 0);
                #pragma omp for schedule(dynamic, 128)
                for(size_t iter = 0; iter < n; iter++){
                    int i = insertion_order[iter];
                    auto curr = greedy_search(nodes[i].vector_data, nodes[i].bitmap, nodes[i].numeric_attr, nodes[i].numeric_attr, L_build, local_visited);

                    omp_set_lock(&node_locks[i].lock);
                    robust_prune(i, curr.second, curr_alpha, header->max_degree);
                    
                    vector<int> current_neighbours(nodes[i].num_neighbours);
                    for(int j = 0; j < nodes[i].num_neighbours; j++){
                        current_neighbours[j] = nodes[i].neighbours[j];
                    }
                    omp_unset_lock(&node_locks[i].lock); 
                    
                    for(int neighbour : current_neighbours){
                        omp_set_lock(&node_locks[neighbour].lock);
                        bool already_neighbour = false;
                        for(int k = 0; k < nodes[neighbour].num_neighbours; k++){
                            if(nodes[neighbour].neighbours[k] == i) { already_neighbour = true; break; }
                        }
                        
                        if(!already_neighbour){
                            if(nodes[neighbour].num_neighbours < header->max_degree){
                                int count = nodes[neighbour].num_neighbours;
                                nodes[neighbour].neighbours[count] = i;
                                nodes[neighbour].num_neighbours = count + 1; 
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
        for(size_t i = 0; i < n; i++) omp_destroy_lock(&node_locks[i].lock);
        delete[] node_locks;
        cout << "[NitroRAG] Index Build Completed Successfully." << endl;
    }


    NitroRAG_Vamana(string filename, size_t n, int medoid){
        fd = open(filename.c_str(), O_RDWR | O_CREAT, 0644);
        if(fd < 0) { cerr << "Failed to open " << filename << endl; exit(1); }
        total_filesize = sizeof(Nitro_IndexHeader) + (n * sizeof(Nitro_DiskNode<n_DIMS, max_degree>));
        if(ftruncate(fd, total_filesize) < 0) { cerr << "ftruncate failed" << endl; exit(1); }
        mapped_data = mmap(nullptr, total_filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if(mapped_data == MAP_FAILED) { cerr << "mmap failed" << endl; exit(1); }
        
        madvise(mapped_data, total_filesize, MADV_RANDOM);
        
        header = static_cast<Nitro_IndexHeader*>(mapped_data);
        nodes = reinterpret_cast<Nitro_DiskNode<n_DIMS, max_degree>*>(static_cast<char*>(mapped_data) + sizeof(Nitro_IndexHeader));
        
        header->num_nodes = n;
        header->max_degree = max_degree;
        header->dimensions = n_DIMS;
        header->global_medoid_id = medoid; 
    }

    ~NitroRAG_Vamana(){
        msync(mapped_data, total_filesize, MS_SYNC);
        munmap(mapped_data, total_filesize);
        close(fd);
    }
};
vector<float> load_fvecs(const string& filename, int& num_vectors, int& dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    in.read((char*)&dim, sizeof(int));
    in.seekg(0, ios::end);
    size_t file_size = in.tellg();
    num_vectors = file_size / ((dim + 1) * 4);
    
    vector<float> data(num_vectors * dim);
    in.seekg(0, ios::beg);
    for (int i = 0; i < num_vectors; i++) {
        in.ignore(4);
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
    in.close();
    return data;
}
float calc_l2_dist(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for(int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

struct QueryFilter {
    uint64_t subset_mask;
    float range_min;
    float range_max;
};


// ===========================================================================
// MAIN BENCHMARK
// ===========================================================================
int main() {
    cout << "\n=======================================================================" << endl;
    cout << "  VECTOR DB BENCHMARK: STD vs. FILTERED vs. NITRORAG      " << endl;
    cout << "=======================================================================\n" << endl;

    int num_vectors, n_DIMS, num_queries, query_dims;
    cout << "[1/8] Loading SIFT1M Base Data & Queries..." << endl;
    vector<float> raw_data = load_fvecs("sift/sift_base.fvecs", num_vectors, n_DIMS);
    vector<float> queries = load_fvecs("sift/sift_query.fvecs", num_queries, query_dims);
    if (n_DIMS != 128) { cerr << "Error: n_DIMS must be 128" << endl; exit(1); }

    // THE ABLATION STUDY: RUN ONCE CORRELATED, ONCE UNCORRELATED
    for (bool IS_CORRELATED : {true, false}) {
        cout << "\n\n#######################################################################" << endl;
        cout << ">>> EXPERIMENT PHASE: LABELS SPATIALLY CORRELATED = " << (IS_CORRELATED ? "TRUE" : "FALSE") << endl;
        cout << "#######################################################################\n" << endl;

        cout << "[2/8] Assigning Attributes and Finding Global Medoid" << endl;
        vector<double> mean_vec(n_DIMS, 0.0);
        for (int i = 0; i < num_vectors; i++) 
            for (int d = 0; d < n_DIMS; d++) mean_vec[d] += raw_data[i * n_DIMS + d];
        for (int d = 0; d < n_DIMS; d++) mean_vec[d] /= num_vectors;

        int global_medoid_id = -1;
        float min_dist_to_mean = 1e30f;
        
        vector<uint64_t> document_labels(num_vectors, 0);
        vector<float> document_ranges(num_vectors, 0.0f);
        
        int ACTIVE_LABELS = 32;
        vector<int> label_centers(ACTIVE_LABELS);
        mt19937 rng(42 + (IS_CORRELATED ? 1 : 0));
        uniform_int_distribution<int> dist_nodes(0, num_vectors - 1);
        uniform_real_distribution<float> dist_range(0.0f, 1000.0f); 

        for(int i = 0; i < ACTIVE_LABELS; i++) label_centers[i] = dist_nodes(rng);

        #pragma omp parallel for
        for (int i = 0; i < num_vectors; i++) {
            float dist_to_mean = 0.0f;
            for (int k = 0; k < n_DIMS; k++) {
                float diff = raw_data[i * n_DIMS + k] - mean_vec[k];
                dist_to_mean += diff * diff;
            }
            #pragma omp critical
            {
                if (dist_to_mean < min_dist_to_mean) { min_dist_to_mean = dist_to_mean; global_medoid_id = i; }
            }
            int best_label = 0;
            if (IS_CORRELATED) {
                float best_dist = 1e30f; 
                for (int l = 0; l < ACTIVE_LABELS; l++) {
                    float d = 0;
                    for(int k = 0; k < n_DIMS; k++) {
                        float diff = raw_data[i * n_DIMS + k] - raw_data[label_centers[l] * n_DIMS + k];
                        d += diff * diff;
                    }
                    if (d < best_dist) { best_dist = d; best_label = l; }
                }
            } else {
                best_label = dist_nodes(rng) % ACTIVE_LABELS;
            }
            
            document_labels[i] = (1ULL << best_label);

            uniform_real_distribution<float> prob(0.0f, 1.0f);
            if(prob(rng) < 0.10f) document_labels[i] |= (1ULL << (dist_nodes(rng) % ACTIVE_LABELS));
            if(prob(rng) < 0.10f) document_labels[i] |= (1ULL << (dist_nodes(rng) % ACTIVE_LABELS));
            
            document_ranges[i] = dist_range(rng);
        }

        cout << "[3/8] Generating Randomized Queries & Recall@10 Ground Truths..." << endl;
        int NUM_TEST_QUERIES = 300;
        int GT_K = 10;
        float RANGE_WIDTH = 10.0f;

        vector<QueryFilter> q_sub(NUM_TEST_QUERIES), q_rng(NUM_TEST_QUERIES), q_mix(NUM_TEST_QUERIES);
        for(int q = 0; q < NUM_TEST_QUERIES; q++) {
            q_sub[q].subset_mask = (1ULL << (dist_nodes(rng)%ACTIVE_LABELS)) | (1ULL << (dist_nodes(rng)%ACTIVE_LABELS));
            float c_rng = dist_range(rng);
            q_rng[q].range_min = c_rng - (RANGE_WIDTH / 2.0f);
            q_rng[q].range_max = c_rng + (RANGE_WIDTH / 2.0f);
            q_mix[q].subset_mask = (1ULL << (dist_nodes(rng)%ACTIVE_LABELS));
            float c_mix = dist_range(rng);
            q_mix[q].range_min = c_mix - (RANGE_WIDTH / 2.0f);
            q_mix[q].range_max = c_mix + (RANGE_WIDTH / 2.0f);
        }

        vector<vector<int>> gt_subset(NUM_TEST_QUERIES), gt_range(NUM_TEST_QUERIES), gt_mixed(NUM_TEST_QUERIES);
        long long total_valid_sub = 0, total_valid_rng = 0, total_valid_mix = 0;

        #pragma omp parallel for reduction(+:total_valid_sub, total_valid_rng, total_valid_mix)
        for (int q = 0; q < NUM_TEST_QUERIES; q++) {
            vector<pair<float, int>> c_sub, c_rng, c_mix;
            for (int i = 0; i < num_vectors; i++) {
                float d = calc_l2_dist(&queries[q * n_DIMS], &raw_data[i * n_DIMS], n_DIMS);

                if ((document_labels[i] & q_sub[q].subset_mask) == q_sub[q].subset_mask) c_sub.push_back({d, i});
                if (document_ranges[i] >= q_rng[q].range_min && document_ranges[i] <= q_rng[q].range_max) c_rng.push_back({d, i});
                if (((document_labels[i] & q_mix[q].subset_mask) == q_mix[q].subset_mask) && 
                    (document_ranges[i] >= q_mix[q].range_min && document_ranges[i] <= q_mix[q].range_max)) {
                    c_mix.push_back({d, i});
                }
            }
            sort(c_sub.begin(), c_sub.end()); sort(c_rng.begin(), c_rng.end()); sort(c_mix.begin(), c_mix.end());

            for(int k = 0; k < min(GT_K, (int)c_sub.size()); k++) gt_subset[q].push_back(c_sub[k].second);
            for(int k = 0; k < min(GT_K, (int)c_rng.size()); k++) gt_range[q].push_back(c_rng[k].second);
            for(int k = 0; k < min(GT_K, (int)c_mix.size()); k++) gt_mixed[q].push_back(c_mix[k].second);

            total_valid_sub += c_sub.size(); total_valid_rng += c_rng.size(); total_valid_mix += c_mix.size();
        }
        
        float avg_sub = (float)total_valid_sub / NUM_TEST_QUERIES;
        float avg_rng = (float)total_valid_rng / NUM_TEST_QUERIES;
        float avg_mix = (float)total_valid_mix / NUM_TEST_QUERIES;

        // ---------------- BUILD INDICES ----------------
        cout << "\n[4/8] BUILDING STANDARD VAMANA INDEX" << endl;
        VAMANA<128, 64> std_index(IS_CORRELATED ? "std_corr.bin" : "std_rand.bin", num_vectors);
        std_index.load_data_and_get_medoid(raw_data.data()); 
        std_index.initialize_random_graph(); 
        auto start_s = chrono::high_resolution_clock::now();
        std_index.build_index(1.2f);
        cout << "Build Time: " << chrono::duration<double>(chrono::high_resolution_clock::now() - start_s).count() << "s\n";

        cout << "\n[5/8] BUILDING FILTERED VAMANA INDEX" << endl;
        Filtered_VAMANA<128, 64> filt_index(IS_CORRELATED ? "filt_corr.bin" : "filt_rand.bin", num_vectors);
        filt_index.load_data_and_get_medoids(raw_data.data(), document_labels.data());
        filt_index.initialize_filtered_random_graph(); 
        auto start_f = chrono::high_resolution_clock::now();
        filt_index.Filtered_build_index(1.2f);
        cout << "Build Time: " << chrono::duration<double>(chrono::high_resolution_clock::now() - start_f).count() << "s\n";

        cout << "\n[6/8] BUILDING NITRORAG (JAG) ENGINE" << endl;
        NitroRAG_Vamana<128, 64> nitro_index(IS_CORRELATED ? "nitro_corr.bin" : "nitro_rand.bin", num_vectors, global_medoid_id);
        #pragma omp parallel for
        for(int i = 0; i < num_vectors; i++) {
            for(int d = 0; d < n_DIMS; d++) nitro_index.nodes[i].vector_data[d] = raw_data[i * n_DIMS + d];
            nitro_index.nodes[i].bitmap = document_labels[i];
            nitro_index.nodes[i].numeric_attr = document_ranges[i];
            nitro_index.nodes[i].num_neighbours = 0;
        }
        nitro_index.compute_global_statistics();
        auto start_n = chrono::high_resolution_clock::now();
        nitro_index.build_index(1.2f);
        cout << "Build Time: " << chrono::duration<double>(chrono::high_resolution_clock::now() - start_n).count() << "s\n";

    // -----------------------------------------------------------------------
    // [7/8] RUNNING BENCHMARKS (RECALL@10)
    // -----------------------------------------------------------------------
    cout << "\n[7/8] RUNNING BENCHMARKS (RECALL@10)\n" << endl;
    vector<int> L_values = {10, 20, 40, 70, 100, 200, 400};

    auto run_benchmark = [&](string title, const vector<QueryFilter>& q_list, const vector<vector<int>>& gt_list, bool use_sub, bool use_rng, float avg_valid) {
        cout << "=== " << title << " (Avg Valid Docs: " << avg_valid << " | Selectivity: " << fixed << setprecision(4) << (avg_valid * 100.0f / num_vectors) << "%) ===" << endl;
        printf("%-8s | %-9s %-8s %-7s | %-9s %-8s %-7s | %-9s %-8s %-7s\n", 
               "L_search", "Std R@10", "Lat(ms)", "Vis", "Filt R@10", "Lat(ms)", "Vis", "Nitro R@10", "Lat(ms)", "Vis");
        cout << "-----------------------------------------------------------------------------------------------------------" << endl;
        
        for (int L_query : L_values) {
            double r10_s = 0, r10_f = 0, r10_n = 0;
            double lat_s = 0, lat_f = 0, lat_n = 0;
            long long vis_s = 0, vis_f = 0, vis_n = 0;
            int v_q_std = 0, v_q_filt = 0, v_q_nitro = 0;

            #pragma omp parallel
            {
                vector<uint8_t> vis(num_vectors, 0);
                
                #pragma omp for
                for (int q = 0; q < NUM_TEST_QUERIES; q++) {
                    int exp_k = min(GT_K, (int)gt_list[q].size());
                    if (exp_k == 0) continue;

                    // --- 1. STANDARD VAMANA ---
                    auto ts1 = chrono::high_resolution_clock::now();
                    auto rs = std_index.greedy_search(std_index.header->entry_point, &queries[q * n_DIMS], L_query, L_query, vis);
                    int h_s = 0, ret_s = 0;
                    for(auto& cand : rs.first) {
                        bool p_sub = !use_sub || ((document_labels[cand.second] & q_list[q].subset_mask) == q_list[q].subset_mask);
                        bool p_rng = !use_rng || (document_ranges[cand.second] >= q_list[q].range_min && document_ranges[cand.second] <= q_list[q].range_max);
                        if(p_sub && p_rng) {
                            if(find(gt_list[q].begin(), gt_list[q].end(), cand.second) != gt_list[q].end()) h_s++;
                            if(++ret_s == GT_K) break;
                        }
                    }
                    auto ts2 = chrono::high_resolution_clock::now();

                    // --- 2. FILTERED VAMANA ---
                    int h_f = 0, ret_f = 0;
                    size_t v_f_size = 0;
                    double l_f = 0;
                    if (use_sub) { 
                        auto tf1 = chrono::high_resolution_clock::now();
                        auto rf = filt_index.Filtered_greedy_search(filt_index.header->filter_medoids, &queries[q * n_DIMS], q_list[q].subset_mask, L_query, L_query, vis);
                        for(auto& cand : rf.first) {
                            bool p_rng = !use_rng || (document_ranges[cand.second] >= q_list[q].range_min && document_ranges[cand.second] <= q_list[q].range_max);
                            if(p_rng) {
                                if(find(gt_list[q].begin(), gt_list[q].end(), cand.second) != gt_list[q].end()) h_f++;
                                if(++ret_f == GT_K) break;
                            }
                        }
                        auto tf2 = chrono::high_resolution_clock::now();
                        l_f = chrono::duration<double, std::milli>(tf2 - tf1).count();
                        v_f_size = rf.second.size();
                    }

                    // --- 3. NITRORAG ---
                    uint64_t q_bit = use_sub ? q_list[q].subset_mask : 0ULL;
                    float q_min = use_rng ? q_list[q].range_min : -1e9f;
                    float q_max = use_rng ? q_list[q].range_max : 1e9f;
                    
                    auto tn1 = chrono::high_resolution_clock::now();
                    auto rn = nitro_index.greedy_search(&queries[q * n_DIMS], q_bit, q_min, q_max, L_query, vis);
                    int h_n = 0, ret_n = 0;
                    for(auto& cand : rn.first) {
                        if (cand.penalty <= 1e-5f) {
                            if(find(gt_list[q].begin(), gt_list[q].end(), cand.id) != gt_list[q].end()) h_n++;
                            if(++ret_n == GT_K) break;
                        }
                    }
                    auto tn2 = chrono::high_resolution_clock::now();

                    // --- CRITICAL AGGREGATION ---
                    #pragma omp critical
                    {
                        lat_s += chrono::duration<double, std::milli>(ts2 - ts1).count();
                        r10_s += (double)h_s / exp_k;
                        vis_s += rs.second.size();
                        v_q_std++;

                        if (use_sub) {
                            lat_f += l_f;
                            r10_f += (double)h_f / exp_k;
                            vis_f += v_f_size;
                            v_q_filt++;
                        }

                        lat_n += chrono::duration<double, std::milli>(tn2 - tn1).count();
                        r10_n += (double)h_n / exp_k;
                        vis_n += rn.second.size();
                        v_q_nitro++;
                    }
                }
            }

            // --- PRINTING RESULTS ---
            if (v_q_nitro > 0) {
                char filt_r10[32], filt_lat[32], filt_vis[32];
                if (use_sub && v_q_filt > 0) {
                    snprintf(filt_r10, sizeof(filt_r10), "%.2f%%", (r10_f / v_q_filt) * 100.0);
                    snprintf(filt_lat, sizeof(filt_lat), "%.3f", lat_f / v_q_filt);
                    snprintf(filt_vis, sizeof(filt_vis), "%lld", vis_f / v_q_filt);
                } else {
                    snprintf(filt_r10, sizeof(filt_r10), "N/A");
                    snprintf(filt_lat, sizeof(filt_lat), "N/A");
                    snprintf(filt_vis, sizeof(filt_vis), "N/A");
                }

                printf("%-8d | %-8.2f%% %-8.3f %-7lld | %-9s %-8s %-7s | %-8.2f%% %-8.3f %-7lld\n",
                    L_query,
                    (v_q_std > 0) ? (r10_s / v_q_std) * 100.0 : 0.0,
                    (v_q_std > 0) ? (lat_s / v_q_std) : 0.0,
                    (v_q_std > 0) ? (vis_s / v_q_std) : 0LL,
                    filt_r10, filt_lat, filt_vis,
                    (r10_n / v_q_nitro) * 100.0, (lat_n / v_q_nitro), (vis_n / v_q_nitro));
            }
        }
        cout << endl;
    };

    run_benchmark("SUBSET FILTER ONLY", q_sub, gt_subset, true, false, avg_sub);
    run_benchmark("RANGE FILTER ONLY", q_rng, gt_range, false, true, avg_rng);
    run_benchmark("MIXED FILTER (SUBSET + RANGE)", q_mix, gt_mixed, true, true, avg_mix);
} 

cout << "\n[8/8] BENCHMARK COMPLETE." << endl;
return 0;
}