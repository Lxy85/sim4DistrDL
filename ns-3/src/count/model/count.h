/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef COUNT_H
#define COUNT_H

#include <string>
#include <vector>

// TODO: What was the point of using boost::scoped_array?
#include <boost/scoped_array.hpp>

namespace ns3 {

// =============================================================================

#ifndef FLAJOLET_MARTIN_H
#define FLAJOLET_MARTIN_H



// =============================================================================

class FlajoletMartin {
public:
  FlajoletMartin();

  double approximateCount(std::vector<std::string>& items, /*const int L,*/ const int M);
  int getM() { return M; }

private:
  int ls1(uint64_t n);
  int ls0(uint64_t n);
  uint64_t vls1(uint64_t n);
  uint64_t vls0(uint64_t n);
  void initialHash(std::vector<std::string>& items, boost::scoped_array<uint64_t>& b, int n);
  double decodeBitString(const boost::scoped_array<uint64_t>& b);

//  int L; // by using 64-bit unsigned integers, we define L = 64
  int M;
  static double PHI;
};

// =============================================================================

#endif // FLAJOLET_MARTIN_H

// =============================================================================

}

#endif /* COUNT_H */

