#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <random>
#include <limits>
#include <fstream>
#include <utility>
#include <thread>
#include <future>

template <typename T>
class matrix;

using Matrixf = matrix<float>;
using Matrixd = matrix<double>;
using Matrixi = matrix<int>;
