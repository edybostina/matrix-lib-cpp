#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

template <typename T>
class matrix;

using Matrixf = matrix<float>;
using Matrixd = matrix<double>;
using Matrixi = matrix<int>;
