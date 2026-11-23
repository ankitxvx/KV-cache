#include "kvcache/scheduler.h"
#include "kvcache/metrics.h"
#include <algorithm>

namespace kvcache {
Scheduler::Scheduler(const ProductionConfig& config) : config_(config) { config_.validate(); }
RequestId Scheduler::add_request(SequenceId seq, int prompt, int generated, int priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    // This scheduler admits whole prompts. Reject oversized prompts explicitly.
    if (!seq.valid() || prompt <= 0 || prompt > config_.max_prefill_tokens || generated <= 0 ||
        waiting_queue_.size() >= static_cast<size_t>(config_.max_waiting_requests) ||
        static_cast<int64_t>(prompt) + generated > std::numeric_limits<int>::max()) return {};
    RequestId id{next_id_++};
    InferenceRequest req;
    req.id=id; req.seq_id=seq; req.prompt_length=prompt; req.max_new_tokens=generated;
    req.priority=priority; req.arrival_time=std::chrono::steady_clock::now();
    requests_.emplace(id.value, req); waiting_queue_.push_back(id);
    ++total_requests_; MetricsRegistry::instance().total_requests.increment(); return id;
}
SchedulerOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerOutput output;
    for (auto id : running_queue_) {
        if (requests_.at(id.value).state == RequestState::DECODING) {
            output.decode_requests.push_back(id); ++output.total_batch_tokens;
        }
    }
    admit_requests(output);
    output.batch_size = static_cast<int>(output.decode_requests.size()+output.prefill_requests.size());
    MetricsRegistry::instance().active_batch_size.set(output.batch_size);
    return output;
}
void Scheduler::admit_requests(SchedulerOutput& output) {
    std::stable_sort(waiting_queue_.begin(), waiting_queue_.end(), [this](auto a, auto b) {
        const auto& ra=requests_.at(a.value); const auto& rb=requests_.at(b.value);
        return ra.priority != rb.priority ? ra.priority > rb.priority : ra.arrival_time < rb.arrival_time;
    });
    int tokens=config_.max_prefill_tokens;
    int64_t budget=int64_t(available_gpu_blocks_)+available_cpu_blocks_;
    for (auto it=waiting_queue_.begin(); it!=waiting_queue_.end() &&
            running_queue_.size()<static_cast<size_t>(config_.max_batch_size);) {
        auto& r=requests_.at(it->value);
        int64_t need=(int64_t(r.prompt_length)+r.max_new_tokens+config_.block_size-1)/config_.block_size;
        if (r.prompt_length>tokens || (budget_set_ && need>budget)) {++it; continue;}
        if (budget_set_) budget-=need;
        tokens-=r.prompt_length; r.state=RequestState::PREFILLING;
        r.start_time=std::chrono::steady_clock::now();
        running_queue_.push_back(r.id); output.prefill_requests.push_back(r.id);
        output.total_batch_tokens+=r.prompt_length; it=waiting_queue_.erase(it);
    }
    if (budget_set_) { available_gpu_blocks_=0; available_cpu_blocks_=static_cast<int>(std::min<int64_t>(budget, INT32_MAX)); }
}
bool Scheduler::can_admit(const InferenceRequest&) const {
    return running_queue_.size()<static_cast<size_t>(config_.max_batch_size);
}
void Scheduler::update(RequestId id, RequestState state) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=requests_.find(id.value); if(it==requests_.end()) return;
    if(state==RequestState::COMPLETED || state==RequestState::ABORTED) {
        if(state==RequestState::COMPLETED) ++completed_count_;
        auto remove=[id](auto& queue){ queue.erase(std::remove(queue.begin(),queue.end(),id),queue.end()); };
        remove(waiting_queue_); remove(running_queue_); requests_.erase(it); return;
    }
    if(it->second.state==RequestState::PREFILLING && state==RequestState::DECODING) it->second.state=state;
}
void Scheduler::on_token_generated(RequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=requests_.find(id.value); if(it==requests_.end()) return;
    auto& r=it->second;
    if(r.state!=RequestState::PREFILLING && r.state!=RequestState::DECODING) return;
    ++r.generated_tokens; r.state=RequestState::DECODING;
    MetricsRegistry::instance().total_tokens_generated.increment();
    if(r.generated_tokens>=r.max_new_tokens) {
        ++completed_count_; running_queue_.erase(std::remove(running_queue_.begin(),running_queue_.end(),id),running_queue_.end());
        requests_.erase(it);
    }
}
void Scheduler::abort(RequestId id) { update(id,RequestState::ABORTED); }
std::optional<InferenceRequest> Scheduler::get_request(RequestId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=requests_.find(id.value); if(it==requests_.end()) return std::nullopt; return it->second;
}
int Scheduler::waiting_count() const {std::lock_guard<std::mutex> lock(mutex_); return static_cast<int>(waiting_queue_.size());}
int Scheduler::running_count() const {std::lock_guard<std::mutex> lock(mutex_); return static_cast<int>(running_queue_.size());}
void Scheduler::set_available_blocks(int gpu,int cpu) {
    if(gpu<0 || cpu<0) throw std::invalid_argument("Negative block budget");
    std::lock_guard<std::mutex> lock(mutex_); available_gpu_blocks_=gpu;available_cpu_blocks_=cpu;budget_set_=true;
}
} // namespace kvcache
