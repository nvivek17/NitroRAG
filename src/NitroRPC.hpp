#pragma once
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

using namespace std;

struct StructuredQuery{
    u_int64_t subset_mask;
    float range_min;
    float range_max;
    vector<float> vec;
};

class NitroRPC{
    public:
    static bool send_all(int sock, const char* buffer, size_t length){
        size_t bytes_sent=0;
        while(bytes_sent< length){
            ssize_t n= send(sock,buffer+bytes_sent,length-bytes_sent,0);
            if(n<=0) return false;
            bytes_sent+=n;
        }
        return true;
    }

    static bool recv_all(int sock,char* buffer, size_t length){
        size_t bytes_received=0;
        while(bytes_received<length){
            ssize_t n=recv(sock,buffer+bytes_received,length-bytes_received,0);
            if(n<=0) return false;
            bytes_received+=n;
        }
        return true;
    }

    static bool send_ints(int sock,const vector<int>& data){
        int sz=data.size();
        if(!send_all(sock,(const char*)&sz,sizeof(int))) return false;
        if(sz>0){
            if(!send_all(sock,(const char*)data.data(),sz*sizeof(int))) return false;
        }
        return true;
    }

    static bool recv_ints(int sock,vector<int>& data){
        int sz=0;
        if(!recv_all(sock,(char*)&sz,sizeof(int))) return false;
        data.resize(sz);
        if(sz>0){
            if(!recv_all(sock,(char*)data.data(),sz*sizeof(int))) return false;
        }
        return true;
    }

    static bool recv_structured_query(int sock, StructuredQuery& query, int expected_dim){
        char meta_buffer[16];
        if(!recv_all(sock,meta_buffer,16)) return false;
        memcpy(&query.subset_mask,meta_buffer,8);
        memcpy(&query.range_min,meta_buffer+8,4);
        memcpy(&query.range_max,meta_buffer+12,4);

        query.vec.resize(expected_dim);
        if(!recv_all(sock,(char*)query.vec.data(),expected_dim*sizeof(float))) return false;
        return true;
    }

};