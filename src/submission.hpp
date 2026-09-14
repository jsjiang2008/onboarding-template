// submission.hpp

#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
    // Store the number of rows in the grid.
    std::size_t rows_;
    // Store the number of columns in the grid.
    std::size_t cols_;
    // Store the grid data in a 1D vector for better cache performance.
    std::vector<double> data_;

public:
    // Constructor to initialize the grid with given rows and columns.
    Grid(std::size_t rows, std::size_t cols)
        : rows_{rows},
          cols_{cols},
          data_(rows * cols, 0.0) {}

    // Accessor for non-const access to grid elements.
    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    // Accessor for const access to grid elements.
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    // Get the number of rows in the grid.
    std::size_t rows() const {
        return rows_;
    }

    // Get the number of columns in the grid.
    std::size_t cols() const {
        return cols_;
    }
};

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    // Get the number of rows and columns from the old grid.
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();

    // Handle edge cases where the grid has no rows or columns.
    if (rows == 0 || cols == 0) {
        return;
    }

    // Copy the top and bottom boundaries.
    for (std::size_t j = 0; j < cols; ++j) {
        new_grid(0, j) = old_grid(0, j);
        new_grid(rows - 1, j) = old_grid(rows - 1, j);
    }

    // Copy the left and right boundaries.
    for (std::size_t i = 1; i + 1 < rows; ++i) {
        new_grid(i, 0) = old_grid(i, 0);
        new_grid(i, cols - 1) = old_grid(i, cols - 1);
    }

    // Apply the five-point stencil to interior cells.
    for (std::size_t i = 1; i + 1 < rows; ++i) {
        for (std::size_t j = 1; j + 1 < cols; ++j) {
            new_grid(i, j) =
                0.5 * old_grid(i, j)
                + 0.125 * (
                    old_grid(i - 1, j)
                    + old_grid(i + 1, j)
                    + old_grid(i, j - 1)
                    + old_grid(i, j + 1)
                );
        }
    }
}