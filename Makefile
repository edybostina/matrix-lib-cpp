CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude

TEST_DIR := tests
EX_DIR   := examples
OBJ_DIR  := build

TEST_MAIN := $(TEST_DIR)/main.cpp
DEMO_MAIN := $(EX_DIR)/demo.cpp

TEST_TARGET := $(OBJ_DIR)/matrix-test
DEMO_TARGET := $(OBJ_DIR)/matrix-demo

all: $(TEST_TARGET) $(DEMO_TARGET)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(TEST_TARGET): $(OBJ_DIR) $(TEST_MAIN)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_MAIN)

$(DEMO_TARGET): $(OBJ_DIR) $(DEMO_MAIN)
	$(CXX) $(CXXFLAGS) -o $@ $(DEMO_MAIN)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

demo: $(DEMO_TARGET)
	./$(DEMO_TARGET)

clean:
	rm -rf $(OBJ_DIR)

.PHONY: all clean test demo