#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

class Grid {
private:
    std::size_t rows_;
    std::size_t cols_;
    // Row-major storage: consecutive columns occupy consecutive addresses
    std::vector<double> data_;

    static std::size_t cell_count(std::size_t rows, std::size_t cols) {
        if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
            throw std::length_error("Grid dimensions overflow");
        }
        return rows * cols;
    }

    friend void apply_stencil(const Grid& old_grid, Grid& new_grid);

public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_{rows}, cols_{cols}, data_(cell_count(rows, cols), 0.0) {}


    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    std::size_t rows() const { 
        return rows_; 
    }
    std::size_t cols() const { 
        return cols_; 
    }
};

// inline keeps this header safe to include from multiple .cpp files
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    if (&old_grid == &new_grid) {
        // Updating in place would overwrite values still needed by neighbors
        throw std::invalid_argument("The stencil needs two distinct grids");
    }
    const std::size_t rows = old_grid.rows_;
    const std::size_t cols = old_grid.cols_;
    if (rows != new_grid.rows_ || cols != new_grid.cols_) {
        throw std::invalid_argument("Grid dimensions must match");
    }
    // The stencil is defined only for grids with at least one interior cell
    if (rows == 0 || cols == 0) {
        return;
    }
    // These grids contain only boundary cells
    if (rows < 3 || cols < 3) {
        std::copy(old_grid.data_.begin(), old_grid.data_.end(), new_grid.data_.begin());
        return;
    }

    const double* old_data = old_grid.data_.data();
    double* new_data = new_grid.data_.data();

    std::copy_n(old_data, cols, new_data);
    const std::size_t bottom = (rows - 1) * cols;
    std::copy_n(old_data + bottom, cols, new_data + bottom);

    // Small grids rarely repay the cost of coordinating multiple workers
    // OpenMP chooses the team size; OMP_NUM_THREADS can tune it per machine
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(old_grid.data_.size() >= 65536)
#endif
    for (std::size_t i = 1; i < rows - 1; ++i) {
        const double* prev = old_data + (i - 1) * cols;
        const double* curr = old_data + i * cols;
        const double* next = old_data + (i + 1) * cols;
        double* out = new_data + i * cols;

        // Each worker owns complete output rows, including their boundaries
        out[0] = curr[0];
        out[cols - 1] = curr[cols - 1];

        // Distinct owning vectors guarantee that output cannot overwrite input
        // No alignment promise: shifted neighbor addresses need not be aligned
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (std::size_t j = 1; j < cols - 1; ++j) {
            out[j] = 0.5 * curr[j] + 0.125 * (prev[j] + next[j] + curr[j - 1] + curr[j + 1]);
        }
    }
}
