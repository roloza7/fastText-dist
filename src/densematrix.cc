/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"

#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <mpi.h>
#include <iostream>
#include <string.h>
#include "utils.h"
#include "vector.h"

#include <mpi.h>

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
#include <immintrin.h>
#endif

namespace fasttext {

DenseMatrix::DenseMatrix() : DenseMatrix(0, 0) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n) : Matrix(m, n), data_(m * n) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
    : Matrix(other.m_, other.n_), data_(std::move(other.data_)) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n, real* dataPtr)
    : Matrix(m, n), data_(dataPtr, dataPtr + (m * n)) {}

void DenseMatrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void DenseMatrix::uniformThread(real a, int block, int32_t seed) {
  std::minstd_rand rng(block + seed);
  std::uniform_real_distribution<> uniform(-a, a);
  int64_t blockSize = (m_ * n_) / 10;
  for (int64_t i = blockSize * block;
       i < (m_ * n_) && i < blockSize * (block + 1);
       i++) {
    data_[i] = uniform(rng);
  }
}

void DenseMatrix::uniform(real a, unsigned int thread, int32_t seed) {
  if (thread > 1) {
    std::vector<std::thread> threads;
    for (int i = 0; i < thread; i++) {
      threads.push_back(std::thread([=]() { uniformThread(a, i, seed); }));
    }
    for (int32_t i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  } else {
    // webassembly can't instantiate `std::thread`
    uniformThread(a, 0, seed);
  }
}

void DenseMatrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void DenseMatrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real DenseMatrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw EncounteredNaNError();
  }
  return std::sqrt(norm);
}

void DenseMatrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

real DenseMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec[j];
  }
  if (std::isnan(d)) {
    throw EncounteredNaNError();
  }
  return d;
}

void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += at(i, j);
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += a * at(i, j);
  }
}

/* Abstract over AVX512F, AVX, and SSE intrinsics, using the one available on this machine. */
#if defined(__AVX512F__)
using Register = __m512;
inline Register Add(Register first, Register second) { return _mm512_add_ps(first, second); }
inline Register Set1(float to) { return _mm512_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm512_mul_ps(first, second); }
#elif defined(__AVX__)
using Register = __m256;
inline Register Add(Register first, Register second) { return _mm256_add_ps(first, second); }
inline Register Set1(float to) { return _mm256_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm256_mul_ps(first, second); }
#elif defined(__SSE__)
using Register = __m128;
inline Register Add(Register first, Register second) { return _mm_add_ps(first, second); }
inline Register Set1(float to) { return _mm_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm_mul_ps(first, second); }
#endif

/* Faster routine for averaging rows of a matrix on x86.
 * The idea here is to keep the accumulators in registers if possible. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
template <unsigned Cols> void averageRowsFast(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix &matrix) {
  // Columns must be a multiple of how many floats fit in a register.
  static_assert(Cols % (sizeof(Register) / 4) == 0);
  constexpr unsigned RegisterCount = Cols / (sizeof(Register) / 4);
  // These should be aligned by aligned.h
  assert(reinterpret_cast<uintptr_t>(x.data()) % sizeof(Register) == 0);
  assert(reinterpret_cast<uintptr_t>(matrix.data()) % sizeof(Register) == 0);

  // Guard against empty list of rows with default NaN behavior.
  if (rows.empty()) {
    x.zero();
    x.mul(1.0 / rows.size());
    return;
  }

  // Copy the first row to accumulation registers.
  Register accum[RegisterCount];
  auto row = rows.cbegin();
  const Register *base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
  for (unsigned i = 0; i < RegisterCount; ++i) {
    accum[i] = base[i];
  }
  // Add the rows after the first.
  for (++row; row != rows.cend(); ++row) {
    base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
    for (unsigned i = 0; i < RegisterCount; ++i) {
      accum[i] = Add(accum[i], base[i]);
    }
  }
  // Multiply by (1.0 / rows.size()) and write to x.
  Register mul = Set1(1.0 / rows.size());
  for (unsigned i = 0; i < RegisterCount; ++i) {
    reinterpret_cast<Register*>(x.data())[i] = Multiply(accum[i], mul);
  }
}
#endif

void DenseMatrix::averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const {
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
  switch (cols()) {
    case 512:
      // Maximum number that can fit all in registers on AVX512F.
      averageRowsFast<512>(x, rows, *this);
      return;
    case 256:
      averageRowsFast<256>(x, rows, *this);
      return;
    case 64:
      averageRowsFast<64>(x, rows, *this);
      return;
    case 32:
      averageRowsFast<32>(x, rows, *this);
      return;
    case 16:
      averageRowsFast<16>(x, rows, *this);
      return;
  }
#endif
  x.zero();
  for (auto it = rows.cbegin(); it != rows.cend(); ++it) {
    addRowToVector(x, *it);
  }
  x.mul(1.0 / rows.size());
}

void DenseMatrix::save(std::ostream& out) const {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = intgemm::AlignedVector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

int DenseMatrix::sync(int tag, real loss) {
  // Adding a second layer of Hogwild! here, on top of the multithreading
  if (!init_mpi_) {
    gather_buffer_ = intgemm::AlignedVector<real>(m_ * n_);
    std::fill(gather_buffer_.begin(), gather_buffer_.end(), 0.0f);
    init_mpi_ = true;
  }

  const uint64_t matrix_size = m_ * n_;

  int world_size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (loss < 0) {
    // Loss is negative is a dummy value to keep nodes from running out of sync.
    std::fill(gather_buffer_.begin(), gather_buffer_.end(), 0.0f);
    MPI_Allreduce(
      MPI_IN_PLACE,
      gather_buffer_.data(),
      matrix_size,
      MPI_FLOAT,
      MPI_SUM,
      MPI_COMM_WORLD);
    // Check if the recv buffer is all -1s, if so, we can skip the rest of the sync.
    bool should_halt = true;
    for (size_t i = 0; i < gather_buffer_.size(); ++i) {
      if (gather_buffer_[i] != 0.0f) {
        should_halt = false;
        break;
      }
    }
    if (should_halt) {
      return 0; // Skip the rest of the sync if all nodes are in sync.
    }
    return 1; // Continue with the sync if any node has data.
  } 

  MPI_Allreduce(
    data_.data(),
    gather_buffer_.data(),
    matrix_size,
    MPI_FLOAT,
    MPI_SUM,
    MPI_COMM_WORLD);

  for (int64_t i = 0; i < matrix_size; i++) {
    // If the received data is NaN or Inf, we skip the interpolation.
    data_[i] = (gather_buffer_[i] / ((float) world_size));
  }

  return 1;

};

  // MPI_Iallreduce(MPI_IN_PLACE, &loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);



  // DEBUG: Wait 100ms
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // // Get random target for interpolation.
  // int r = rand() % (world_size);
  // r = (r == rank) ? (r + 1) % world_size : r; // Avoid self-interaction.


  // // 1. check sends_ (array of MPI_Request) is used to track the latest send operation
  // uint64_t i;
  // for (i = 0; i < std::min(world_size, 4); i++) {
  //   // Check for first completed or cancelled request, ans signal the spot is free.
  //   if (sends_[i] != MPI_REQUEST_NULL) {
  //     int flag;
  //     int err = MPI_Test(&sends_[i], &flag, MPI_STATUS_IGNORE);
  //     if (err != MPI_SUCCESS) {
  //       // Send failed, (likely due to node exit)
  //       sends_[i] = MPI_REQUEST_NULL; // Mark the request as completed.
  //       break; // Exit the loop since this is now a free spot.
  //     }


  //     if (flag) {
  //       MPI_Wait(&sends_[i], MPI_STATUS_IGNORE);
  //       sends_[i] = MPI_REQUEST_NULL; // Mark the request as completed.
  //       break;
  //     }
  //     continue; // If not completed, continue to the next request.
  //   }
  //   break; // Found a free spot.
  // }

  // // If we didn't find a free spot, we don't send anything.
  // if (i < std::min(world_size, 4)) {
  //   // Prepare the send buffer with the data and loss value.
  //   // i * matrix_size + matrix_size is the position for the loss value. (we will only send that slice)
  //   memcpy(send_buffer_.data() + i * (matrix_size + 1), data_.data(), matrix_size * sizeof(real));
    
  //   // std::cout << "Send Buffer Size: " << send_buffer_.size() << ", Matrix Size: " << matrix_size << std::endl;
  //   // std::cout << "Buffer pointer index: " << i * (matrix_size + 1) << std::endl;
  //   // std::cout << "Loss pointer index: " << i * (matrix_size + 1) + matrix_size << std::endl;
  //   send_buffer_[i * (matrix_size + 1) + matrix_size] = loss; // Store the loss value at the end of the slice.


  //   MPI_Isend(send_buffer_.data() + i * (matrix_size + 1), matrix_size + 1, MPI_FLOAT, r, tag, MPI_COMM_WORLD, &sends_[i]);
  //   // MPI_Isend(data_.data(), matrix_size, MPI_FLOAT, r, tag, MPI_COMM_WORLD, &sends_[i]);
  // }
  // if (recv_request_ != MPI_REQUEST_NULL) {
  //   // If we have a pending receive request, check if it is completed.
  //   int flag;
  //   MPI_Status status;
  //   int err = MPI_Test(&recv_request_, &flag, &status);
  //   if (err != MPI_SUCCESS) {
  //     // Receive failed, (likely due to node exit)
  //     recv_request_ = MPI_REQUEST_NULL; // Mark the request as completed.
  //     return; // Exit the function since we can't proceed.
  //   }
    
  //   if (flag) {
  //     int count;
  //     MPI_Get_count(&status, MPI_FLOAT, &count);
  //     if (count != matrix_size + 1) {
  //       std::cout << "Received data size: " << count << ", expected size: " << matrix_size << std::endl;
  //       throw std::runtime_error("Received data size does not match expected size.");
  //     }
  //     // Interpolate the received data.
  //     MPI_Wait(&recv_request_, MPI_STATUS_IGNORE);
  //     real other_loss = recv_buffer_[matrix_size]; // The last element is the loss value.

  //     // Interpolate towards lower loss.
  //     if (std::isnan(other_loss) || std::isinf(loss)) {
  //       std::cerr << "Received NaN or Inf loss value: " << other_loss << std::endl;
  //       other_loss = 0.0; // Treat NaN or Inf as zero loss.
  //     }
  //     t_ = loss < other_loss ? 0.0 : (other_loss / (loss + other_loss));

  //     for (int64_t j = 0; j < matrix_size; j++) {
  //       data_[j] = recv_buffer_[j] * (1.0 - t_) + data_[j] * t_;
  //       if (std::isnan(data_[j]) || std::isinf(data_[j])) {
  //         std::cerr << "Encountered NaN or Inf in interpolated data at index " << j << std::endl;
  //       }
  //     }
  //     // Reset the receive request.
  //     recv_request_ = MPI_REQUEST_NULL;
  //   }
  // } else {
  //   MPI_Irecv(recv_buffer_.data(), matrix_size + 1, MPI_FLOAT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_request_);
  // }

} // namespace fasttext
