// compile: /std:c++17
// baseline_min_ms: 300
// baseline_max_ms: 1500
// weight: 0.3
// description: Simple Windows MSVC compilation test

#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    auto doubled = numbers;
    std::transform(doubled.begin(), doubled.end(), doubled.begin(), 
                   [](int n) { return n * 2; });
    
    std::cout << "Original: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Doubled: ";
    for (int n : doubled) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    return 0;
}