#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

class TestCase {
public:
    TestCase(const std::string& group, const std::string& name)
        : group_(group), name_(name) {}
    virtual ~TestCase() = default;

    virtual void run() = 0;

    std::string getGroup() const { return group_; }
    std::string getName() const { return name_; }

private:
    std::string group_;
    std::string name_;
};

class TestRegistry {
public:
    static TestRegistry& getInstance() {
        static TestRegistry instance;
        return instance;
    }

    void registerTest(TestCase* test) {
        tests_.push_back(test);
    }

    int runAllTests() {
        int passed = 0;
        int failed = 0;

        std::cout << "Running " << tests_.size() << " tests..." << std::endl;

        for (auto* test : tests_) {
            std::cout << "[ RUN      ] " << test->getGroup() << "." << test->getName() << std::endl;
            try {
                test->run();
                std::cout << "[       OK ] " << test->getGroup() << "." << test->getName() << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << "[  FAILED  ] " << test->getGroup() << "." << test->getName() << std::endl;
                std::cout << "             " << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cout << "[  FAILED  ] " << test->getGroup() << "." << test->getName() << std::endl;
                std::cout << "             Unknown exception" << std::endl;
                failed++;
            }
        }

        std::cout << "\n========================================================" << std::endl;
        std::cout << "Total Tests: " << (passed + failed) << std::endl;
        std::cout << "Passed:      " << passed << std::endl;
        std::cout << "Failed:      " << failed << std::endl;
        
        return (failed == 0) ? 0 : 1;
    }

private:
    TestRegistry() = default;
    std::vector<TestCase*> tests_;
};

struct TestRegistrar {
    TestRegistrar(TestCase* test) {
        TestRegistry::getInstance().registerTest(test);
    }
};

#define ASSERT_TRUE(condition) \
    if (!(condition)) throw std::runtime_error("Assertion failed: " #condition)

#define ASSERT_EQ(a, b) \
    if (std::abs((double)(a) - (double)(b)) > 1e-9) \
        throw std::runtime_error("Assertion failed: expected " + std::to_string(a) + ", got " + std::to_string(b))


#define TEST_CLASS_NAME(group, name) group##_##name##_Test
#define TEST_REGISTRAR_NAME(group, name) group##_##name##_Registrar

#define TEST(group, name)                                                       \
    class TEST_CLASS_NAME(group, name) : public TestCase {                      \
    public:                                                                     \
        TEST_CLASS_NAME(group, name)() : TestCase(#group, #name) {}             \
        void run() override;                                                    \
    };                                                                          \
                                                                                \
    static TEST_CLASS_NAME(group, name) *test_instance_##group##_##name =       \
        new TEST_CLASS_NAME(group, name)();                                     \
                                                                                \
    static TestRegistrar TEST_REGISTRAR_NAME(group, name)(                      \
        test_instance_##group##_##name);                                        \
                                                                                \
    void TEST_CLASS_NAME(group, name)::run()
