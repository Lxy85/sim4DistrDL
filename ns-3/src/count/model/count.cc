/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "count.h"


#include <cmath>

#include "farmhash.h"

namespace ns3 {

// =============================================================================

// Implements the Flajolet-Martin algorithm for approximately counting unique
// items in a data stream,
// Flajolet and Martin, "Probabilistic Counting Algorithms for Data Base Applications",
// Journal of Computer and System Sciences 31, 182-209 (1985).



// =============================================================================

double FlajoletMartin::PHI = 0.77351;

FlajoletMartin::FlajoletMartin() {}

// Return the position of the least significant 1 bit in n (= count trailing zeros)
// (not used - included for completeness)
int FlajoletMartin::ls1(uint64_t n) {
  // note __builtin_ctz() takes unsigned int, not long, so use __builtin_ctzl()
  return (n == 0 ? 8*sizeof(n) : __builtin_ctzl(n));
}

// Return the position of the least significant 0 bit in n (= count trailing ones)
int FlajoletMartin::ls0(uint64_t n) {
  // note __builtin_ctz() takes unsigned int, not long, so use __builtin_ctzl()
  return (n == 0 ? 8*sizeof(n) : __builtin_ctzl(~n));
}

// Return 2^(position of the least significant 1 bit in n)
uint64_t FlajoletMartin::vls1(uint64_t n) {
  return (n & (-n));
}

// Return 2^(position of the least significant 0 bit in n)
// (not used - included for completeness)
uint64_t FlajoletMartin::vls0(uint64_t n) {
  return ((~n) & (n + 1));
}

// Universal hashing: the point is to have a different hash function for each seed m
uint64_t hash(std::string s, int m) {
  return util::Hash64WithSeed(s.c_str(), s.length(), m);
}

void FlajoletMartin::initialHash(std::vector<std::string>& items, boost::scoped_array<uint64_t>& b, const int n) {
  uint64_t h;
  for (int m = 0; m < M; m++) {
    for (int i = 0; i < n; i++) {
      h = hash(items[i], m);
      b[i*M + m] = vls1(h);
    }
  }
}

double FlajoletMartin::decodeBitString(const boost::scoped_array<uint64_t>& b) {
//  phiAdj <- phi*(1 + 0.31/M)
  double R = 0;
  for (int m = 0; m < M; m++) {
    R += ls0(b[m]);
  }
  R /= M;
  // the form of the following does not seem to make much difference...
  return pow(2.0, R)/PHI;
  // or its floor?
//  return floor(pow(2.0, R)/PHI);
}

// Returns the approximate word count of items, averaged over M trials.
double FlajoletMartin::approximateCount(std::vector<std::string>& items, /*const int L,*/ const int M) {
//  this->L = L;
  this->M = M;

  // initial random bit strings
  int numTotalItems = items.size();
  boost::scoped_array<uint64_t> b(new uint64_t[numTotalItems*M]);
  boost::scoped_array<uint64_t> bb(new uint64_t[M]);

  initialHash(items, b, numTotalItems);

  // approximate number of items
  for (int m = 0; m < M; m++) {
    bb[m] = 0;
    for (int i = 0; i < numTotalItems; i++) {
      bb[m] |= b[i*M + m];
    }
  }

  double nApprox = decodeBitString(bb);
  return nApprox;
}

// =============================================================================


}

