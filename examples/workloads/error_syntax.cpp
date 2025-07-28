// compile: -std=c++17
// baseline_min_ms: 50
// baseline_max_ms: 200
// weight: 0.3
// description: Syntax error compilation failure

#include <iostream>

int main() {
    std::cout << "This has a syntax error" << std::endl
    // Missing semicolon above
    return 0;
}