// compile: -O2 -std=c++17
// baseline_min_ms: 100
// baseline_max_ms: 500
// weight: 0.1
// description: Simple baseline test without any libraries

#include <iostream>

int main() {
    std::cout << "Simple hello world baseline!" << std::endl;
    return 0;
}