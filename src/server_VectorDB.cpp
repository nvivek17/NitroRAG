#pragma once

#include <iostream>
#include <thread>
#include <netinet/in.h>
#include "JAG_new.cpp"
#include "NitroRPC.hpp"

using namespace std;
const int K=5;
NitroRAG_Vamana<128,64>* db_index;

void handle_db_client(int client_sock){
    StructuredQuery sq;
    if(NitroRPC::recv_structured_query(client_sock,sq,128)){
        cout<<"[VectorDB] Query Received. Subset Filters:"<<sq.subset_mask<<" | Range filter : ["<<sq.range_min<<","<<sq.range_max<<"]"<<endl;
        vector<uint8_t> vis(db_index->header->num_nodes,0);
        auto result=db_index->greedy_search(sq.vec.data(),sq.subset_mask,sq.range_min,sq.range_max,100,vis);

        vector<int> received_tokens;
        for(int i=0;i<result.first.size()&&i<K;i++){
            int curr_node_id=result.first[i].id;
            //convert vectors back into tokens
            received_tokens.push_back(db_index->nodes[curr_node_id].vector_data);
        }
        NitroRPC::send_ints(client_sock,received_tokens);
    }
    close(client_sock);
}
int main(){
    cout<<"Loading NitroRAG Vector Database..."<< endl;
    db_index=new NitroRAG_Vamana<128,64>("Retrieval/nitrorag_index.bin",1000000,-1);
    int server_fd=socket(AF_INET,SOCK_STREAM,0);
    int opt=1;setsockopt(server_fd,SOL_SOCKET,SO_REUSEADDR,&opr,sizeof(opt));
    struct sockaddr_in address;
    address.sin_family=AF_INET;
    address.sin_addr.s_addr=INADDR_ANY;
    address.sin_port=htons(8080);
    bind(server_fd,(struct sockaddr*)&address,sizeof(address));
    listen(server_fd,100);
    cout<<"[Vector DB] Online. Listening on Port 8080..."<<endl;
    while(true){
        int client_socket=accept(server_fd,nullptr,nullptr);
        std::thread(handle_db_client,client_socket).detach();
    }
    return 0;
}
