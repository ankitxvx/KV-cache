#include "kvcache/logger.h"

namespace kvcache {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::~Logger() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void Logger::set_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.close();
    }
    if (!path.empty()) {
        file_.open(path, std::ios::app);
    }
}

std::string Logger::format_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

void Logger::log(LogLevel level, const char* component, const std::string& message) {
    message_count_.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << "[" << format_timestamp() << "] "
        << "[" << log_level_name(level) << "] "
        << "[" << component << "] "
        << message;

    std::string formatted = oss.str();

    std::lock_guard<std::mutex> lock(mutex_);

    if (console_enabled_) {
        std::cerr << formatted << "\n";
    }

    if (file_.is_open()) {
        file_ << formatted << "\n";
        // Flush on WARN+ for reliability
        if (static_cast<uint8_t>(level) >= static_cast<uint8_t>(LogLevel::WARN)) {
            file_.flush();
        }
    }
}

} // namespace kvcache
