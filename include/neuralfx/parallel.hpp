// A small fixed pool of worker threads for data-parallel loops whose results must not depend on the number of threads
// (study N: the 3D simulator, the volume renderer and the volume runtime; docs/STUDY_3D.md).
//
// run(n, fn) calls fn(i) once for every i in [0, n), on the pool's workers and the calling thread, and returns when all
// have finished. Each index must write its own outputs and read only what no other index of the same call writes; then
// the result is the same on any number of threads, whatever order the indices run in. The workers are created by the
// constructor (or resize()) and wait on a condition variable between calls: run() never allocates.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace nfx {

class Pool {
 public:
  explicit Pool(int threads = 1) { resize(threads); }
  ~Pool() { stop(); }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  // Threads that work on a run(): the workers and the caller.
  int threads() const { return static_cast<int>(workers_.size()) + 1; }

  // Set-up only (allocates): `threads` in all, the caller included; 0 or less means one per hardware thread.
  void resize(int threads) {
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    stop();
    {
      std::lock_guard lock(mutex_);
      stopping_ = false;
    }
    workers_.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) workers_.emplace_back([this] { work(); });
  }

  template <class F>
  void run(int n, F&& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    using Fn = std::remove_reference_t<F>;
    {
      std::lock_guard lock(mutex_);
      call_ = [](void* c, int i) { (*static_cast<Fn*>(c))(i); };
      ctx_ = const_cast<void*>(static_cast<const void*>(&fn));
      n_ = n;
      next_.store(0, std::memory_order_relaxed);
      busy_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    wake_.notify_all();
    drain();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return busy_ == 0; });
  }

 private:
  void drain() {
    for (int i = next_.fetch_add(1, std::memory_order_relaxed); i < n_; i = next_.fetch_add(1, std::memory_order_relaxed)) call_(ctx_, i);
  }
  void work() {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
      }
      drain();
      {
        std::lock_guard lock(mutex_);
        if (--busy_ == 0) done_.notify_one();
      }
    }
  }
  void stop() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    workers_.clear();  // joins
  }

  std::mutex mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::jthread> workers_;
  void (*call_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0};
  int busy_ = 0;
  std::uint64_t generation_ = 0;
  bool stopping_ = false;
};

// run() on a pool, or serially without one.
template <class F>
void parallel_for(Pool* pool, int n, F&& fn) {
  if (pool) {
    pool->run(n, fn);
  } else {
    for (int i = 0; i < n; ++i) fn(i);
  }
}

}  // namespace nfx
