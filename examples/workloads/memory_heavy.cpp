// compile: -O2 -std=c++17
// baseline_min_ms: 450
// baseline_max_ms: 2500
// weight: 0.6
// description: Large static array allocations and deep template instantiation

#include <iostream>
#include <array>

constexpr size_t LARGE_SIZE = 10000;

struct LargeStruct {
    std::array<double, 1000> data;
    int id;
    
    LargeStruct(int i) : id(i) {
        for (size_t j = 0; j < data.size(); ++j) {
            data[j] = static_cast<double>(i * 1000 + j);
        }
    }
};

template<int N>
struct DeepNesting {
    template<typename T>
    struct Inner {
        template<int M>
        struct VeryInner {
            static constexpr int value = N + M + DeepNesting<N-1>::template Inner<T>::template VeryInner<M-1>::value;
        };
        
        template<>
        struct VeryInner<0> {
            static constexpr int value = N;
        };
    };
    
    static constexpr int compute() {
        return Inner<int>::template VeryInner<10>::value;
    }
};

template<>
struct DeepNesting<0> {
    template<typename T>
    struct Inner {
        template<int M>
        struct VeryInner {
            static constexpr int value = M;
        };
    };
    
    static constexpr int compute() {
        return Inner<int>::template VeryInner<10>::value;
    }
};

int main() {
    static std::array<int, LARGE_SIZE> large_array;
    static std::array<LargeStruct, 100> struct_array{
        LargeStruct{0}, LargeStruct{1}, LargeStruct{2}, LargeStruct{3}, LargeStruct{4},
        LargeStruct{5}, LargeStruct{6}, LargeStruct{7}, LargeStruct{8}, LargeStruct{9}
    };
    
    // Initialize large array
    for (size_t i = 0; i < LARGE_SIZE; ++i) {
        large_array[i] = static_cast<int>(i * i % 1000);
    }
    
    // Process data
    long long sum = 0;
    for (const auto& val : large_array) {
        sum += val;
    }
    
    for (const auto& s : struct_array) {
        sum += s.id;
        for (size_t i = 0; i < 10; ++i) {
            sum += static_cast<long long>(s.data[i]);
        }
    }
    
    constexpr int deep_result = DeepNesting<20>::compute();
    sum += deep_result;
    
    std::cout << "Large array processing sum: " << sum << std::endl;
    
    return 0;
}