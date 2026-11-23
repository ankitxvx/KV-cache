#pragma once

#include "types.h"
#include <string>
#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <atomic>

namespace kvcache {

/// Log severity levels.
enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
    OFF   = 6,
};

inline const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        case LogLevel::OFF:   return "OFF  ";
    }
    return "?????";
}

inline LogLevel log_level_from_string(const std::string& s) {
    if (s == "TRACE" || s == "trace") return LogLevel::TRACE;
    if (s == "DEBUG" || s == "debug") return LogLevel::DEBUG;
    if (s == "INFO"  || s == "info")  return LogLevel::INFO;
    if (s == "WARN"  || s == "warn")  return LogLevel::WARN;
    if (s == "ERROR" || s == "error") return LogLevel::ERROR;
    if (s == "FATAL" || s == "fatal") return LogLevel::FATAL;
    if (s == "OFF"   || s == "off")   return LogLevel::OFF;
    return LogLevel::INFO;
}

/// Thread-safe structured logger with console and file sinks.
class Logger {
public:
    /// Get the singleton logger instance.
    static Logger& instance();

    /// Set minimum log level. Messages below this level are discarded.
    void set_level(LogLevel level) { min_level_ = level; }
    LogLevel level() const { return min_level_; }

    /// Set file output. Empty string disables file logging.
    void set_file(const std::string& path);

    /// Set whether to log to console (stderr). Default: true.
    void set_console(bool enabled) { console_enabled_ = enabled; }

    /// Log a message at the given level with a component tag.
    void log(LogLevel level, const char* component, const std::string& message);

    /// Check if a level would be logged (for avoiding expensive formatting).
    bool should_log(LogLevel level) const {
        return static_cast<uint8_t>(level) >= static_cast<uint8_t>(min_level_.load());
    }

    /// Get total messages logged.
    int64_t message_count() const { return message_count_.load(); }

private:
    Logger() = default;
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string format_timestamp() const;

    std::atomic<LogLevel> min_level_{LogLevel::INFO};
    std::atomic<bool>     console_enabled_{true};
    std::mutex            mutex_;
    std::ofstream         file_;
    std::atomic<int64_t>  message_count_{0};
};

// ---------------------------------------------------------------------------
// Logging macros — compile-time level filtering
// ---------------------------------------------------------------------------

#define KVCACHE_LOG(level, component, msg)                                     \
    do {                                                                        \
        auto& _logger = ::kvcache::Logger::instance();                         \
        if (_logger.should_log(level)) {                                       \
            _logger.log(level, component, msg);                                \
        }                                                                       \
    } while (0)

#define KVCACHE_LOG_TRACE(component, msg)  KVCACHE_LOG(::kvcache::LogLevel::TRACE, component, msg)
#define KVCACHE_LOG_DEBUG(component, msg)  KVCACHE_LOG(::kvcache::LogLevel::DEBUG, component, msg)
#define KVCACHE_LOG_INFO(component, msg)   KVCACHE_LOG(::kvcache::LogLevel::INFO,  component, msg)
#define KVCACHE_LOG_WARN(component, msg)   KVCACHE_LOG(::kvcache::LogLevel::WARN,  component, msg)
#define KVCACHE_LOG_ERROR(component, msg)  KVCACHE_LOG(::kvcache::LogLevel::ERROR, component, msg)
#define KVCACHE_LOG_FATAL(component, msg)  KVCACHE_LOG(::kvcache::LogLevel::FATAL, component, msg)

} // namespace kvcache
