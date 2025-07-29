// compile: -O0 -std=c++17
// baseline_min_ms: 500
// baseline_max_ms: 2000
// weight: 1.0
// description: Simple Hello World program

#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}