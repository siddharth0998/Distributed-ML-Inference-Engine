#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

template <typename T>
class SafeQueue {
private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_shutdown = false;

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(std::move(item));
        m_cv.notify_one();
    }

    // Standard pop: waits forever until an item arrives
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return !m_queue.empty() || m_shutdown; });
        
        if (m_queue.empty() && m_shutdown) return false;

        item = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    // NEW: Timed pop for dynamic batching
    template <typename Rep, typename Period>
    bool pop_for(T& item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(m_mutex);
        bool success = m_cv.wait_for(lock, timeout, [this]() { return !m_queue.empty() || m_shutdown; });
        
        if (!success || (m_queue.empty() && m_shutdown)) return false;

        item = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_shutdown = true;
        m_cv.notify_all();
    }
};