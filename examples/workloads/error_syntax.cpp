// compile: -std=c++17
// baseline_min_ms: 350
// baseline_max_ms: 1600
// weight: 0.3
// description: Syntax error compilation failure

#include <iostream>

int main() {
    std::cout << "This has a syntax error" << std::endl
    // Missing semicolon above
    return 0;
}