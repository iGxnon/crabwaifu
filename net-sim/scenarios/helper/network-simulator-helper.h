#ifndef _NETWORK_SIMULATOR_HELPER_H
#define _NETWORK_SIMULATOR_HELPER_H

#include "ns3/node.h"

using namespace ns3;

class NetworkSimulatorHelper {
public:
  NetworkSimulatorHelper();
  void Run(Time);
  Ptr<Node> GetLeftNode() const;
  Ptr<Node> GetRightNode() const;

private:
  void RunSynchronizer() const;
  Ptr<Node> left_node_, right_node_;
};

#endif /* _NETWORK_SIMULATOR_HELPER_H */
