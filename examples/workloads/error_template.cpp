// compile: -std=c++17
// baseline_min_ms: 350
// baseline_max_ms: 3800
// weight: 0.5
// description: Template instantiation error

#include <iostream>

template<typename T>
void process(T value) {
    // This will cause an error for types that don't support this method
    value.nonexistent_method();
}

int main() {
    process(42);  // This will cause a template instantiation error
    return 0;
}