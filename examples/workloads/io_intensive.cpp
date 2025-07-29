// compile: -O1 -std=c++17
// baseline_min_ms: 1600
// baseline_max_ms: 6000
// weight: 0.7
// description: Many standard library includes and large generated output

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <locale>
#include <regex>
#include <atomic>
#include <queue>
#include <stack>
#include <deque>
#include <list>
#include <forward_list>
#include <array>
#include <tuple>
#include <utility>
#include <type_traits>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

struct DetailedInfo {
    std::string name;
    int id;
    double value;
    std::vector<int> data;
    
    DetailedInfo(const std::string& n, int i, double v) 
        : name(n), id(i), value(v), data(100, i) {}
    
    void process() {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<int>(value * i + id);
        }
    }
};

class ComplexProcessor {
private:
    std::vector<DetailedInfo> items;
    
public:
    ComplexProcessor() {
        for (int i = 0; i < 1000; ++i) {
            items.emplace_back("Item_" + std::to_string(i), i, i * 3.14159);
        }
    }
    
    void processAll() {
        for (auto& item : items) {
            item.process();
        }
    }
    
    double calculateSum() const {
        double sum = 0.0;
        for (const auto& item : items) {
            sum += item.value;
            for (int val : item.data) {
                sum += val * 0.001;
            }
        }
        return sum;
    }
};

int main() {
    std::vector<std::string> data = {"hello", "world", "test"};
    std::map<std::string, int> counts;
    
    for (const auto& str : data) {
        counts[str] = static_cast<int>(str.length());
    }
    
    ComplexProcessor processor;
    processor.processAll();
    
    double result = processor.calculateSum();
    
    std::cout << "String lengths:" << std::endl;
    for (const auto& [key, value] : counts) {
        std::cout << key << ": " << value << std::endl;
    }
    
    std::cout << "Complex processing result: " << result << std::endl;
    
    return 0;
}