# sim4DistrD

1. Description

   Sim4DistrDL is a novel discrete event simulator, which includes a deep learning module and a network simulation module to facilitate simulation of DNN-based distributed applications. Sim4DistrDL incorporates OpenNN[1] and other deep learning libs into the extensively used discrete event simulator, NS-3. NS-3[2] is a process-based discrete-event simulation framework. The detailed information of NS3 can be found at https://www.nsnam.org/. Some popular DNN models, e.g. MLP, LeNet and VGG, have been implemented on Sim4DistrDL. 
   
   To achieve a personalized environment to simulate distributed learning, users also can easily include new deep learning models or libs into Sim4DistrDL by following  the instruction below and those have been implemented on Sim4DistrDL. Meanwhile, we will extend sim4DistrDL to new application scenarios.
  
2. Environment

   OS:Ubuntu 18.04
   
   boost>=1.48
   
   G++>=7.4
   
   python>=2.7

3. Prerequisite

   To compile Sim4DistrDL, the following packages must be installed in the environment.

   Boost libraries should be installed on the system:

       sudo aptitude install libboost-all-dev

   In order to run visualizer module, the following should be installed:

       sudo apt-get install python-dev python-pygraphviz python-kiwi

       sudo apt-get install python-pygoocanvas python-gnome2

       sudo apt-get install python-rsvg ipython
   
4. Getting started

   Federated learning (FL) is emerging as a new paradigm to train machine learning (ML) models in a distributed environment. We present an application case of the decentralized federated learning on sim4DistrDL. CFA is a fully distributed (or serverless) federated learning approach. it leverages the cooperation of devices that perform learning inside the network by iterating local training and then mutual interactions via consensus-based aggregation. 

Step 1: Construct a topology
      
  Use the Create() function in the NodeContainer class to create the required nodes
	
  NodeContainer nodes; 
	
  nodes.Create (6);  
	
  NodeContainer n0=NodeContainer(nodes.Get(0),nodes.Get(1));
	
            ……
					 
  NetDeviceContainer devices; 
	
  NetDeviceContainer d0 = pointToPoint.Install (n0);
	
            ……         
			     
  Ipv4InterfaceContainer interfaces = address.Assign (d0); 
	
            ……


Step 2: Implement and deploy cache (LRU-based Caching) for nodes and initialize the cache size for each node

        1) Configuration : 
     
           Configure wscript file, add “./utils/create-module.py src/Lru”, to construct a Lru module;
       
        2) Implement Lru module
     
           Cache data: LRUCache::set(int key, string value)  
        
           a. judge whether the number of the cached data has reached the maximum capacity of the cache 
        
           b. If it has reached the maximum capacity, delete the last data in the linked list and store the new data at the head of the linked list. 
        
           c. Otherwise, the new received data is directly stored at the head of the linked list.
        
           Remove the cached data : void LRUCache::remove(CacheNode *node)
      
           a. the last data in the linked list will be deleted
       
           Look up a specific data:  string LRUCache::get(int key)
      
           a. the value of the found data is returned. 
     
           b. the data is placed at the head of the linked list.
       
       3)  Compile LRU-based Caching component to make the simulator include Lru module
    
           ./waf configure
     
           ./waf build
     
          The configure file contains the shared libraries and header files used for compiling files, and the path that must be entered when compiling scripts with a             waf tool.

Step 3: Edge servers collect data from end devices 

        Simulator::Schedule(Seconds(t),&send1,Send_sock1,training_data);  

                                 ……
                                 
Step 4: Including deep learning lib 

        Include all the header files that support deep learning into a header file. 
   
        In wscript file, configure cross compiling of network simulation module and deep learning module in the simulator.
        
        def build(bld):
        
 	         bld.stlib(“deep{\_}learning”)
            
  	         module.uselib = ‘deep{\_}learning’
            
 	         module.source = ‘deep{\_}learning/**/*.cpp’
            
	         module.full{\_}headers = ‘deep{\_}learning/**/*.cpp’
            
            ./waf configure
            
            ./waf builds
            
Step 5: Implement and Deploy of MLP model at edge servers as an application

        Firstly, we should include the header file (deep.h ) of the deep learning module in the simulation. we should encapsulate the model into an application of NS3         to facilitate neural network training
        
                          ……
                          
        #include "/home/allen/ndnSIM/ns-3/src/ndnSIM/opennn/deep.h“
                          ……
                          
         class MyApp : public Application{  
         
                          ……
                          
                }
                
          void MyApp::StartApplication (void){
          
                          ……
                          
             training_strategy.perform_training();   
             
                         …….      
                             
            };
                   
Then, the training parameters of the model are configured, such as the maximum training round, the threshold to guarantee model converged, etc. 

Install the application at the edge servers

                     Ptr<MyApp> app1 = CreateObject<MyApp> ();
                     
                     nodes.Get (2)->AddApplication (app1); 
                     
                       ……
                       
Step 6: Implement the transfer of model parameters

                class MyApp : public Application {
                 
                 private:
                           virtual void StartApplication (void);
                           
                                      ……
                                      
                           void SendPacket (Tensor<type,1>  str);
                           
                                   ……
                                   
                 };

                 void MyApp::StartApplication (void){
                 
                     m_socket->Bind ();
                       
                     m_socket->Connect (m_peer);
                       
                                  ……
                                          
                     training_strategy.perform_training();
                      
                     Tensor<type,1>  client1_pararms;
                     
                     client1_pararms=neural_network.get_parameters();
                     
                     SendPacket (client1_pararms);
                     
}

Step 7: Start the application to train MLP

          app->SetStartTime (Seconds (1.));
          
                        ……
                        
Step 8: Implement the parameter aggregation at the Edge server

     class MyApp1: public Application {
     
      public:
                        ……
                        
            void aggregation(void);
            
                        ……
                        
       }:
      
      
Reference

[1]  R Lopez. Opennn: An open source neural networks c++ library. Artificial Intelligence Techniques, Ltd.: Salamanca, Spain, 20

[2]  ns3 development team. ns3 network simulator. https://www.nsnam.org/,2000

