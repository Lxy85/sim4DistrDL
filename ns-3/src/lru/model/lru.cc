/**
 * System design: LRU cache (least recently used).
 *
 * cpselvis(cpselvis@gmail.com)
 * Oct 8th, 2016
 */
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>

#include "lru.h"

using namespace std;
using namespace ns3;


  CacheNode::CacheNode(int k, string v)
  {
    key=k;
    value=v;
    pre=NULL;
    next=NULL; 
  }
  CacheNode::CacheNode(int k, Sample_0 s)
  {
    key=k;
    sample=s;
    pre=NULL;
    next=NULL; 
  }


  LRUCache::LRUCache(int capacity)
  {
    size = capacity;
    head = NULL;
    tail = NULL;
  }

 string LRUCache::get(int key)
  {
    map<int, CacheNode *>::iterator it = mp.find(key);
    if (it != mp.end())
    {
      CacheNode *node = it -> second;
      remove(node);
      setHead(node);
      return node -> value;
    }
    else
    {
      return NULL;
    }
  }
  Sample_0 LRUCache::get(int key,int m)
  {
    Sample_0 s={NULL,NULL,0,0,0};
    map<int, CacheNode *>::iterator it = mp.find(key);
    if (it != mp.end())
    {
      CacheNode *node = it -> second;
      remove(node);
      setHead(node);
      return node -> sample;
    }
   else
    {
      return s;
    }
  }

  /*void LRUCache::set(int key, Ptr<Packet> value)
  {
    map<int, CacheNode *>::iterator it = mp.find(key);
    if (it != mp.end())
    {
      CacheNode *node = it -> second;
      node -> value = value;
      remove(node);
      setHead(node);
    }
    else
    {
      CacheNode *newNode = new CacheNode(key, value);
      if (mp.size() >= size)
      {
	map<int, CacheNode *>::iterator iter = mp.find(tail -> key);
      	remove(tail);
	mp.erase(iter);
      }
      setHead(newNode);
      mp[key] = newNode;
    }
  }*/
  void LRUCache::set(int key, string value)
  {
    map<int, CacheNode *>::iterator it = mp.find(key);
    if (it != mp.end())
    {
      CacheNode *node = it -> second;
      node -> value = value;
      remove(node);
      setHead(node);
    }
    else
    {
      CacheNode *newNode = new CacheNode(key, value);
      if (mp.size() >= size)
      {
        map<int, CacheNode *>::iterator iter = mp.find(tail -> key);
        remove(tail);
        mp.erase(iter);
      }
      setHead(newNode);
      mp[key] = newNode;
    }
  }

  void LRUCache::set(int key, Sample_0 sample)
  {
    map<int, CacheNode *>::iterator it = mp.find(key);
    if (it != mp.end())
    {
      CacheNode *node = it -> second;
      node -> sample = sample;
      remove(node);
      setHead(node);
    }
    else
    {
      CacheNode *newNode = new CacheNode(key, sample);
      if (mp.size() >= size)
      {
        map<int, CacheNode *>::iterator iter = mp.find(tail -> key);
        remove(tail);
        mp.erase(iter);
      }
      setHead(newNode);
      mp[key] = newNode;
    }
  }


  void LRUCache::remove(CacheNode *node)
  {
    if (node -> pre != NULL)
    {
      node -> pre -> next = node -> next;
    }
    else
    {
      head = node -> next;
    }
    if (node -> next != NULL)
    {
      node -> next -> pre = node -> pre;
    }
    else
    {
      tail = node -> pre;
    }
  }

  void LRUCache::setHead(CacheNode *node)
  {
    node -> next = head;
    node -> pre = NULL;

    if (head!= NULL)
    {
      head -> pre = node;
    }
    head = node;
    if (tail == NULL)
    {
      tail = head;
    }
  }

