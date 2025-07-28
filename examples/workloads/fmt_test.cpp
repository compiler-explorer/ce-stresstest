// compile: -O2 -std=c++17
// lib: fmt/trunk
// baseline_min_ms: 100
// baseline_max_ms: 500
// weight: 0.5
// description: Simple test using fmt library for formatting

#include <iostream>
#include <fmt/format.h>

int main() {
    // Test fmt library functionality
    std::string formatted = fmt::format("Hello, {}! The answer is {}.", "world", 42);
    std::cout << formatted << std::endl;
    
    // More complex formatting
    std::cout << fmt::format("Pi is approximately {:.2f}", 3.14159) << std::endl;
    std::cout << fmt::format("Binary: {:b}, Hex: {:x}", 255, 255) << std::endl;
    
    return 0;
}