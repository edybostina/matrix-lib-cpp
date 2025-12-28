#include "test_framework.hpp"

int main() {
    return TestRegistry::getInstance().runAllTests();
}