#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace hybrid {
struct request_ticket {
    std::string client;
    std::string session;
    bool critical;
};

// Rotate clients first to prevent starvation. Within a client, prefer critical work without bypassing session FIFO.
inline request_ticket * next_request(const std::vector<request_ticket *> & queue, const std::string & last_client) {
    std::vector<request_ticket *> eligible;
    for (size_t i = 0; i < queue.size(); ++i) {
        bool first = true;
        for (size_t j = 0; j < i; ++j) {
            if (queue[i]->client == queue[j]->client && queue[i]->session == queue[j]->session) { first = false; break; }
        }
        if (first) {
            eligible.push_back(queue[i]);
        }
    }
    request_ticket * after = nullptr;
    request_ticket * wrap = nullptr;
    for (auto * ticket : eligible) {
        if (!wrap || ticket->client < wrap->client) { wrap = ticket; }
        if (ticket->client > last_client && (!after || ticket->client < after->client)) { after = ticket; }
    }
    auto * selected = after ? after : wrap;
    if (selected) {
        for (auto * ticket : eligible) {
            if (ticket->client == selected->client && ticket->critical && !selected->critical) { selected = ticket; }
        }
    }
    return selected;
}

class request_scheduler {
public:
    class lease {
    public:
        explicit lease(request_scheduler & scheduler, bool active = true) : scheduler(scheduler), active(active) {}
        lease(const lease &) = delete;
        explicit operator bool() const { return active; }
        ~lease() {
            if (!active) { return; }
            std::lock_guard<std::mutex> lock(scheduler.mutex);
            scheduler.busy = false;
            scheduler.changed.notify_all();
        }
    private:
        request_scheduler & scheduler;
        bool active;
    };

    lease enter(const std::string & client, const std::string & session, bool critical, const std::function<bool()> & disconnected) {
        request_ticket ticket{client, session, critical};
        std::unique_lock<std::mutex> lock(mutex);
        queue.push_back(&ticket);
        while (!disconnected() && (busy || next_request(queue, last_client) != &ticket)) {
            changed.wait_for(lock, std::chrono::milliseconds(50));
        }
        queue.erase(std::find(queue.begin(), queue.end(), &ticket));
        if (disconnected()) {
            changed.notify_all();
            return lease(*this, false);
        }
        busy = true;
        last_client = client;
        return lease(*this);
    }
private:
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<request_ticket *> queue;
    std::string last_client;
    bool busy = false;
};
}
