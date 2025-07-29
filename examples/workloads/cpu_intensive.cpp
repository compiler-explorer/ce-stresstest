// compile: -O2 -std=c++20
// lib: ctre/trunk
// baseline_min_ms: 1700
// baseline_max_ms: 6000
// weight: 0.8
// description: CPU-intensive compile-time regex processing with CTRE

#include <iostream>
#include <string_view>
#include <array>
#include <ctre.hpp>

// Simple regex patterns that are CPU-intensive but won't break the compiler
constexpr auto email_pattern = ctre::match<R"([a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]{2,4})">;
constexpr auto number_pattern = ctre::match<R"(\d+)">;
constexpr auto word_pattern = ctre::match<R"([a-zA-Z]+)">;
constexpr auto url_pattern = ctre::match<R"(https://[a-zA-Z0-9]+\.[a-zA-Z0-9]+)">;  // Fixed: removed ? and .-

// Test data for compile-time processing
constexpr std::array test_data = {
    std::string_view{"user@example.com"},
    std::string_view{"test@domain.org"},
    std::string_view{"https://example.com"},
    std::string_view{"https://test.org"},
    std::string_view{"12345"},
    std::string_view{"hello"},
    std::string_view{"world"},
    std::string_view{"invalid-email"},
    std::string_view{"bad-url"},
    std::string_view{"999999"}
};

// Compile-time validation function
template<size_t N>
constexpr int count_matches(const std::array<std::string_view, N>& data) {
    int email_count = 0;
    int number_count = 0;
    int word_count = 0;
    int url_count = 0;
    
    for (const auto& item : data) {
        if (email_pattern(item)) email_count++;
        if (number_pattern(item)) number_count++;
        if (word_pattern(item)) word_count++;
        if (url_pattern(item)) url_count++;
    }
    
    return email_count + number_count + word_count + url_count;
}

// Heavy compile-time computation
constexpr int total_matches = count_matches(test_data);

// Additional compile-time regex processing
template<std::string_view const& str>
constexpr bool validate_string() {
    return email_pattern(str) || number_pattern(str) || word_pattern(str) || url_pattern(str);
}

// Compile-time string validation
constexpr std::string_view test1 = "user@test.com";
constexpr std::string_view test2 = "12345";
constexpr std::string_view test3 = "hello";
constexpr std::string_view test4 = "https://example.com";

constexpr bool result1 = validate_string<test1>();
constexpr bool result2 = validate_string<test2>();
constexpr bool result3 = validate_string<test3>();
constexpr bool result4 = validate_string<test4>();

int main() {
    std::cout << "=== CTRE Compile-Time Regex Results ===" << std::endl;
    std::cout << "Total compile-time matches: " << total_matches << std::endl;
    
    std::cout << "\nIndividual validation results:" << std::endl;
    std::cout << "test1 (" << test1 << "): " << std::boolalpha << result1 << std::endl;
    std::cout << "test2 (" << test2 << "): " << std::boolalpha << result2 << std::endl;
    std::cout << "test3 (" << test3 << "): " << std::boolalpha << result3 << std::endl;
    std::cout << "test4 (" << test4 << "): " << std::boolalpha << result4 << std::endl;
    
    // Runtime processing for additional CPU load
    int runtime_matches = 0;
    
    std::cout << "\n=== Runtime Validation ===" << std::endl;
    for (const auto& item : test_data) {
        bool matched = false;
        
        if (email_pattern(item)) {
            std::cout << "Email: " << item << std::endl;
            matched = true;
        } else if (number_pattern(item)) {
            std::cout << "Number: " << item << std::endl;
            matched = true;
        } else if (word_pattern(item)) {
            std::cout << "Word: " << item << std::endl;
            matched = true;
        } else if (url_pattern(item)) {
            std::cout << "URL: " << item << std::endl;
            matched = true;
        }
        
        if (matched) runtime_matches++;
    }
    
    std::cout << "\nRuntime matches: " << runtime_matches << std::endl;
    
    // Verify compile-time computation worked
    static_assert(total_matches > 0, "Should have found some matches");
    static_assert(result1, "test1 should match");
    static_assert(result2, "test2 should match");
    static_assert(result3, "test3 should match");
    static_assert(result4, "test4 should match");
    
    return 0;
}