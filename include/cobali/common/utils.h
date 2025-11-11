#pragma once

#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace cobali {
namespace utils {

// Logging utilities
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }
    
    void setLogLevel(LogLevel level) {
        log_level_ = level;
    }
    
    void log(LogLevel level, const std::string& message) {
        if (level < log_level_) return;
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG:   level_str = "DEBUG"; break;
            case LogLevel::INFO:    level_str = "INFO"; break;
            case LogLevel::WARNING: level_str = "WARN"; break;
            case LogLevel::ERROR:   level_str = "ERROR"; break;
        }
        
        std::cout << "[" << std::put_time(std::localtime(&time), "%H:%M:%S") 
                  << "] [" << level_str << "] " << message << std::endl;
    }
    
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    
private:
    Logger() : log_level_(LogLevel::INFO) {}
    LogLevel log_level_;
};

// Timing utilities
inline double getElapsedMs(const std::chrono::time_point<std::chrono::steady_clock>& start,
                           const std::chrono::time_point<std::chrono::steady_clock>& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double getElapsedSeconds(const std::chrono::time_point<std::chrono::steady_clock>& start,
                                const std::chrono::time_point<std::chrono::steady_clock>& end) {
    return std::chrono::duration<double>(end - start).count();
}

// String formatting
template<typename... Args>
std::string format(const char* fmt, Args... args) {
    size_t size = std::snprintf(nullptr, 0, fmt, args...) + 1;
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, fmt, args...);
    return std::string(buf.get(), buf.get() + size - 1);
}

} // namespace utils
} // namespace cobali

