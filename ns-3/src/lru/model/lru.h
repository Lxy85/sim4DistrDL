#include <iostream>
#include <string>
#include <cstring>
#include <time.h>
#include <map>
#include <omp.h>
#include "ns3/packet.h"
#include "/home/allen/ndnSIM/ns-3/src/ndnSIM/opennn/sample.h"
using namespace std;
using namespace ns3;
#ifndef LRUCache_H_H  
#define LRUCache_H_H  

class CacheNode {
  public:
  int key;
  string value;
  Sample_0 sample;
  CacheNode *pre, *next;
  CacheNode(int k, string v);
  CacheNode(int k, Sample_0 s);
};


class LRUCache{
private:
                     // Maximum of cachelist size.
  CacheNode *head, *tail;
  map<int, CacheNode *> mp;          // Use hashmap to store
public:
  LRUCache(int capacity);
  int size;  
  string get(int key);
  Sample_0 get(int key,int m);

  void set(int key, string value);
  void set(int key, Sample_0 sample);

  void remove(CacheNode *node);

  void setHead(CacheNode *node);
  
};
#endif 

