#include "kvcache/cache_manager.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace kvcache;
using Clock = std::chrono::steady_clock;
static volatile double checksum = 0;
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}
static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(),values.end());
    return values[static_cast<size_t>(fraction*(values.size()-1))];
}
static void emit(const char* workload, const char* operation, int repetition,
                 int seqs,int length,int layers,size_t bytes,int operations,double us,
                 const std::vector<double>& samples, int blocks) {
    std::cout << workload << ',' << operation << ',' << repetition << ',' << seqs << ','
              << length << ',' << layers << ',' << operations << ',' << bytes << ','
              << us << ',' << operations*1e6/us << ',' << bytes/(us*1000.0) << ','
              << (samples.empty()?0:percentile(samples,.50)) << ','
              << (samples.empty()?0:percentile(samples,.95)) << ','
              << (samples.empty()?0:percentile(samples,.99)) << ',' << blocks << '\n';
}
static void run(const char* name,int sequences,int length,int repetition) {
    KVCacheConfig cfg;
    cfg.num_layers=4; cfg.num_kv_heads=4; cfg.head_dim=64;cfg.block_size=16;
    cfg.max_blocks=sequences*((length+15)/16)+16;
    cfg.eviction_policy=EvictionType::NONE; cfg.dtype=DataType::FP32;
    KVCacheManager manager(cfg);
    size_t elements=size_t(cfg.num_kv_heads)*cfg.head_dim;
    std::vector<float> key(elements),value(elements),out_key(elements),out_value(elements);
    for(size_t i=0;i<elements;++i) {key[i]=float(i)*.125f;value[i]=-float(i)*.25f;}
    std::vector<const void*> keys(cfg.num_layers,key.data()),values(cfg.num_layers,value.data());
    for(int s=0;s<sequences;++s) require(manager.create_sequence({s})==Status::OK,"create failed");
    int operations=sequences*length;
    size_t payload=size_t(operations)*cfg.num_layers*cfg.kv_pair_bytes_per_token();
    std::vector<double> timings;timings.reserve(operations);
    auto begin=Clock::now();
    for(int s=0;s<sequences;++s) for(int t=0;t<length;++t) {
        auto start=Clock::now();auto status=manager.append_kv_all_layers({s},keys,values);
        timings.push_back(elapsed(start));require(status==Status::OK,"append failed");
    }
    double append_us=elapsed(begin);
    int blocks=manager.get_stats().used_blocks;
    require(blocks==sequences*((length+15)/16),"unexpected physical block count");
    if(repetition>=0) emit(name,"append_all_layers",repetition,sequences,length,cfg.num_layers,
        payload,operations,append_us,timings,blocks);

    timings.clear();timings.reserve(size_t(operations)*cfg.num_layers);
    begin=Clock::now();
    for(int s=0;s<sequences;++s) for(int t=0;t<length;++t) for(int l=0;l<cfg.num_layers;++l) {
        auto start=Clock::now();
        auto status=manager.copy_kv({s},l,t,out_key.data(),out_value.data(),elements*sizeof(float));
        timings.push_back(elapsed(start));require(status==Status::OK,"read failed");
        checksum+=out_key[t%elements]+out_value[t%elements];
    }
    double read_us=elapsed(begin);
    require(out_key==key && out_value==value,"retrieved data mismatch");
    if(repetition>=0) emit(name,"copy_kv_layer",repetition,sequences,length,cfg.num_layers,
        payload,operations*cfg.num_layers,read_us,timings,blocks);

    timings.clear();constexpr int copies=32;
    begin=Clock::now();
    for(int i=0;i<copies;++i) {
        auto start=Clock::now();auto status=manager.fork_sequence({0},{1000+i});
        timings.push_back(elapsed(start));require(status==Status::OK,"fork failed");
    }
    double fork_us=elapsed(begin);
    require(manager.get_stats().used_blocks==blocks,"fork copied physical KV blocks");
    if(repetition>=0) emit(name,"fork_sequence",repetition,sequences,length,cfg.num_layers,
        0,copies,fork_us,timings,manager.get_stats().used_blocks-blocks);
    // Check an actual write detaches the shared tail and leaves the source intact.
    auto original=manager.get_num_tokens({0});
    require(manager.append_kv_all_layers({1000},keys,values)==Status::OK,"fork append failed");
    require(manager.get_num_tokens({0})==original,"fork modified source length");
    for(int i=0;i<copies;++i) manager.free_sequence({1000+i});

    std::vector<int> tokens(length);std::iota(tokens.begin(),tokens.end(),0);
    manager.register_prefix({0},tokens.data(),length);
    timings.clear();double match_us=0;
    for(int i=0;i<100;++i) {
        manager.create_sequence({2000});auto start=Clock::now();
        int matched=manager.try_prefix_match({2000},tokens.data(),length);
        double us=elapsed(start);match_us+=us;timings.push_back(us);
        require(matched==length && manager.get_num_tokens({2000})==length,"prefix attachment failed");
        require(manager.copy_kv({2000},3,length-1,out_key.data(),out_value.data(),elements*sizeof(float))==Status::OK,
                "prefix data missing");
        require(out_key==key && out_value==value,"prefix data mismatch");
        manager.free_sequence({2000});
    }
    if(repetition>=0) emit(name,"prefix_attach",repetition,sequences,length,cfg.num_layers,
        0,100,match_us,timings,blocks);
    manager.clear_prefix_cache();
    for(int s=0;s<sequences;++s) manager.free_sequence({s});
    require(manager.get_stats().used_blocks==0,"block leak after workload");
}
int main() {
    try {
        std::cout<<std::fixed<<std::setprecision(6);
        std::cout<<"workload,operation,repetition,sequences,context_tokens,layers,operations,payload_bytes,elapsed_us,operations_per_second,payload_GB_per_second,p50_us,p95_us,p99_us,physical_blocks\n";
        for(int repetition=-1;repetition<7;++repetition) {
            run("8x256",8,256,repetition);
            run("4x1024",4,1024,repetition);
            run("1x4096",1,4096,repetition);
        }
    } catch(const std::exception& e) {std::cerr<<"BENCHMARK INVALID: "<<e.what()<<'\n';return 1;}
}
