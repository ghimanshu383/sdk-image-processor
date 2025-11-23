//
// Created by ghima on 04-11-2025.
//

#ifndef OSFEATURENDKDEMO_UTILITY_H
#define OSFEATURENDKDEMO_UTILITY_H

#include <vector>

namespace ip {
    class Utility {
    public:
        static std::vector<std::vector<float>> generate_gaussian_kernel(int radius, float sigma) {
            int size = 2 * radius + 1;
            std::vector<std::vector<float>> kernel(size, std::vector<float>(size));
            float sum = 0.0;
            float s = 2.0f * sigma * sigma;

            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    float value = std::exp(-(x * x + y * y) / s) / (3.14159f * s);
                    kernel[x + radius][y + radius] = value;
                    sum += value;
                }
            }
            for (auto &row: kernel) {
                for (auto &x: row) {
                    x = x / sum;
                }
            }
            return kernel;
        }
    };
}
#endif //OSFEATURENDKDEMO_UTILITY_H
