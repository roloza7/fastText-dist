/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <array>
#include <mpi.h> // Ensure this is included before any use of MPI_Request

#include "aligned.h"
#include "matrix.h"
#include "real.h"

namespace fasttext {

class Vector;

class DenseMatrix : public Matrix {
 protected:
  intgemm::AlignedVector<real> data_;
  void uniformThread(real, int, int32_t);

  // Interpolation factor for MPI
  real t_ = 0.9;
  bool init_mpi_ = false; // Flag to check if MPI is initialized
  // Requests for asynchronous communication
  std::array<MPI_Request, 4> sends_; // MPI_Request is now a complete type
  intgemm::AlignedVector<real> recv_buffer_; // Buffers for receiving data
  intgemm::AlignedVector<real> send_buffer_; // Buffer for sending loss values

  real loss_recv_buffer_ = 0.0; // Buffer for receiving loss values
  MPI_Request recv_request_; // Request for receiving data
  int latest_send_ = 0; // Index of the latest queried buffer

 public:
  DenseMatrix();
  explicit DenseMatrix(int64_t, int64_t);
  explicit DenseMatrix(int64_t m, int64_t n, real* dataPtr);
  DenseMatrix(const DenseMatrix&) = default;
  DenseMatrix(DenseMatrix&&) noexcept;
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix& operator=(DenseMatrix&&) = delete;
  virtual ~DenseMatrix() noexcept override = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }

  inline const real& at(int64_t i, int64_t j) const {
    assert(i * n_ + j < data_.size());
    return data_[i * n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * n_ + j];
  };

  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }
  void zero();
  void uniform(real, unsigned int, int32_t);

  void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1);
  void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1);

  real l2NormRow(int64_t i) const;
  void l2NormRow(Vector& norms) const;

  real dotRow(const Vector&, int64_t) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i) const override;
  void addRowToVector(Vector& x, int32_t i, real a) const override;
  void averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
  void sync(int, real) override;

  class EncounteredNaNError : public std::runtime_error {
   public:
    EncounteredNaNError() : std::runtime_error("Encountered NaN.") {}
  };
};
} // namespace fasttext
