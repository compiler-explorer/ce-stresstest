// compile: -O2 -std=c++17
// lib: boost/187
// baseline_min_ms: 4000
// baseline_max_ms: 8000
// weight: 0.3
// description: Heavy compile-time load from Boost library headers

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> words = {"hello", "world", "boost", "compilation"};
    
    // Use boost::algorithm
    boost::algorithm::to_upper(words[0]);
    
    // Use boost::format
    boost::format fmt("Processing %1% items with boost!");
    std::cout << fmt % words.size() << std::endl;
    
    // Use boost::regex
    boost::regex pattern("\\w+");
    if (boost::regex_match(words[0], pattern)) {
        std::cout << "Pattern matched!" << std::endl;
    }
    
    return 0;
}