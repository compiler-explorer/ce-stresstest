// compile: /std:c++17
// baseline_min_ms: 3000
// baseline_max_ms: 6000
// weight: 0.3
// description: Windows MSVC compilation test with Windows-specific features

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <chrono>
#include <numeric>

// Use some MSVC-specific features
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996) // Disable deprecation warnings
#endif

// Template metaprogramming that MSVC handles differently than GCC/Clang
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// SFINAE example that tests MSVC's template engine
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
process_integral(T value) {
    return value * 2;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
process_integral(T value) {
    return value * 1.5;
}

// Class with MSVC-specific optimizations
class MSVCOptimizedClass {
private:
    std::vector<int> data_;
    
public:
    MSVCOptimizedClass(size_t size) : data_(size) {
        std::iota(data_.begin(), data_.end(), 1);
    }
    
    // __forceinline is MSVC-specific
    #ifdef _MSC_VER
    __forceinline
    #else
    inline
    #endif
    int compute_sum() const {
        return std::accumulate(data_.begin(), data_.end(), 0);
    }
    
    // Test MSVC's move semantics implementation
    MSVCOptimizedClass(MSVCOptimizedClass&& other) noexcept 
        : data_(std::move(other.data_)) {}
    
    MSVCOptimizedClass& operator=(MSVCOptimizedClass&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }
};

// Lambda with capture that tests MSVC's lambda implementation
auto create_multiplier(int factor) {
    return [factor](int value) -> int {
        return value * factor;
    };
}

int main() {
    try {
        // Test constexpr evaluation
        constexpr int fact5 = Factorial<5>::value;
        std::cout << "5! = " << fact5 << std::endl;
        
        // Test SFINAE overload resolution
        auto int_result = process_integral(42);
        auto float_result = process_integral(3.14);
        std::cout << "Processed: " << int_result << ", " << float_result << std::endl;
        
        // Test MSVC-optimized class
        MSVCOptimizedClass optimizer(1000);
        int sum = optimizer.compute_sum();
        std::cout << "Sum: " << sum << std::endl;
        
        // Test move semantics
        MSVCOptimizedClass moved = std::move(optimizer);
        
        // Test lambda
        auto multiply_by_3 = create_multiplier(3);
        std::cout << "Lambda result: " << multiply_by_3(10) << std::endl;
        
        // Test smart pointers (different implementations between compilers)
        auto ptr = std::make_unique<std::vector<int>>(100, 42);
        std::cout << "Smart pointer size: " << ptr->size() << std::endl;
        
        // Test chrono (implementation varies)
        auto start = std::chrono::high_resolution_clock::now();
        volatile int dummy = 0;
        for (int i = 0; i < 10000; ++i) {
            dummy += i;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Loop took: " << duration.count() << " microseconds" << std::endl;
        
        std::cout << "MSVC compilation test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif