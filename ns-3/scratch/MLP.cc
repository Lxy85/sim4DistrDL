
#include <stdlib.h>
#include <cstdio>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "/home/allen/ndnSIM/ns-3/src/ndnSIM/opennn/deep.h"
#include <utility>

#include <iomanip>
using namespace ns3;
using namespace std;
using namespace OpenNN;
float param=0;
int node_size[3];
int epoch=0;
double client1[5];
char kind1[10];
int size0,size1,size2;
string s0;
int tensor_size1=0;
int tensor_size2=0;
int tensor_size3=0;
Tensor<type,2> data1(145,8);
Tensor<type,2> data2(145,8);
Tensor<type,2> data3(145,8);
Tensor<type,2> data4(145,8);
Tensor<Index, 1> input_indices;
Tensor<Index, 1> target_indices;

Tensor<type,1> client1_pararms,client11_pararms(43),c1;//上一轮训练的参数
Tensor<type,1> client2_pararms,client22_pararms(43),c2;
Tensor<type,1> client3_pararms,client33_pararms(43),c3;
//解析后的数据 
Tensor<type,1> client1_pararms0(43);//client1上一轮训练的参数 
Tensor<type,1> client0_pararms1(43);//client2 上一轮训练的参数 
Tensor<type,1> client2_pararms1(43);//client3上一轮训练的参数 
Tensor<type,1> client1_pararms2(43);

int t=1;
int flag1=1,flag2=1,flag3=1,flag4=1;


string s1,s2,s3;

static float number[43];


NodeContainer nodes;
NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

void send(Ptr<Socket> sock,string str);
void RecvString1(Ptr<Socket> sock);
void RecvString2(Ptr<Socket> sock);
void RecvString3(Ptr<Socket> sock);
void RecvString4(Ptr<Socket> sock);
void change(string s,float number[]);
DataSet data_set("/home/allen/Data/iris.data", ',', 0);


class MyApp : public Application 
{
public:
  void trainning(void);
  //void set_p(void);
  //int number[42];
};
void MyApp::trainning(void)
{
       
   cout << "Client1 OpenNN. Iris Plant Example." << endl;

    srand(static_cast<unsigned>(time(nullptr)));

   
    DataSet data(data1);
   
    data.set_input_target_columns(input_indices,target_indices);


    const Index input_variables_number1 = data_set.get_input_variables_number();
    const Index target_variables_number1 = data_set.get_target_variables_number();
    //const Index input_variables_number1 = 4;
   // const Index target_variables_number1 = 3;
  

    //const Index input_variables_number = data.get_input_variables_number();
    //const Index target_variables_number = data.get_target_variables_number();

    // Neural network

    const Index hidden_neurons_number = 3;
    Tensor<Index, 1> architecture(3);
    architecture[0] = input_variables_number1;
    architecture[1] = 5;
    architecture[2] = target_variables_number1;
    
    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

    /****************set parameters***************/

    Tensor<type,1> avg_pararms(43),a;//平均之后的 


    if(epoch>0)
    {
      //cout<<"client1_pararms0:"<<endl<<client1_pararms0<<endl;
       // ofstream fin("/home/allen/Data/1.txt");
     /*   ifstream inFile1("/home/allen/Data/1.txt",ios::in);
        if(!inFile1)
        {
          cout<<"打开文件失败！"<<endl;
          exit(1);
        }
        string buffer;
        string temp="";
        int j_0=0,k=0;
        while(getline(inFile1,buffer))
        {
          cout<<"()))))))))))"<<endl<<buffer<<endl;
            for(int i=0;i<buffer.size();i++)
            {  
                  if(buffer[i]!=';')
                  { 

                    temp=temp+buffer[i];
                  }
                  if(buffer[i]==';')
                  {
                    
                      number[k]=stof(temp);
                      k++;
                      temp=""; 
                 }
                 if(i==buffer.size()-2)
                 {
                    number[k]=stof(temp);

                 }

           }
           for(int i=0;i<43;i++)
           {
               client1_pararms0(i)=number[i];
           }

            
        }*/
    
    //  cout<<"client1_pararms0:"<<endl<<client1_pararms0<<endl;
      for(int i=0;i<43;i++)
      {
         avg_pararms(i)=client11_pararms(i)+(client1_pararms0(i)-client11_pararms(i))*0.5*0.001;
      }
     // cout<<client1_pararms0<<endl;
      //cout<<"*)))))))))"<<endl;
     // cout<<client11_pararms<<endl;
     // cout<<"**********"<<endl;
      //cout<<avg_pararms<<endl;
      //a=client1_pararms+(client2_pararms-client1_pararms)*c1.constant(0.5f)*c1.constant(0.1f);

      neural_network.set_parameters(avg_pararms);
       // neural_network.set_parameters(a);
    }
    // Training strategy

    TrainingStrategy training_strategy(&neural_network, &data);
    //TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    //training_strategy.set_display_period(10);
    training_strategy.set_maximum_epochs_number(1);

    

    training_strategy.perform_training1();

    // Testing analysis

    //const TestingAnalysis testing_analysis(&neural_network, &data_set);
    const TestingAnalysis testing_analysis(&neural_network, &data);

    const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
 
    client1_pararms=neural_network.get_parameters();
    c1=neural_network.get_parameters();
    //cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$$client1"<<endl;
    //cout<<client1_pararms<<endl;
    s1="";

    for(int i=0;i<43;i++)
    {

       client11_pararms(i)=client1_pararms(i);
       std::stringstream ss;
    
       ss << client1_pararms(i);
       std::string asString = ss.str();
       s1+=asString;
       s1+=';';
    }
  
  
  

    cout << "\nConfusion matrix:\n" << confusion << endl;
    
 
}
class MyApp1 : public Application 
{
public:
  void trainning(void);
};
void MyApp1::trainning(void)
{

    cout << "Client2  Iris Plant Example." << endl;

    srand(static_cast<unsigned>(time(nullptr)));

   
    DataSet data(data2);
   
    data.set_input_target_columns(input_indices,target_indices);


    const Index input_variables_number1 = data_set.get_input_variables_number();
    const Index target_variables_number1 = data_set.get_target_variables_number();
    
   // const Index target_variables_number1 = 3;
  

    //const Index input_variables_number = data.get_input_variables_number();
    //const Index target_variables_number = data.get_target_variables_number();

    // Neural network

    const Index hidden_neurons_number = 3;
    Tensor<Index, 1> architecture(3);
    architecture[0] = input_variables_number1;
    architecture[1] = 5;
    architecture[2] = target_variables_number1;
    
    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

    /****************set parameters***************/
    Tensor<type,1> avg_pararms(43),a;//平均之后的 
    if(epoch>0)
    {

      for(int i=0;i<43;i++)
      {
         avg_pararms(i)=client22_pararms(i)+(client0_pararms1(i)-client22_pararms(i))*0.33*0.001+(client2_pararms1(i)-client22_pararms(i))*0.33*0.001;
      }
      //  a=client2_pararms+(client1_pararms-client2_pararms)*c2.constant(0.33f)*c2.constant(0.1f)+(client3_pararms-client2_pararms)*c2.constant(0.33f)*c2.constant(0.1f);
     // }  
       //neural_network.set_parameters(a);
      
       neural_network.set_parameters(avg_pararms);
    }
    // Training strategy

    TrainingStrategy training_strategy(&neural_network, &data);
    //TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    //training_strategy.set_display_period(10);
    training_strategy.set_maximum_epochs_number(1);


    /**********************trainning************************/

    training_strategy.perform_training1();

    // Testing analysis

    //const TestingAnalysis testing_analysis(&neural_network, &data_set);
    const TestingAnalysis testing_analysis(&neural_network, &data);

    const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

     client2_pararms=neural_network.get_parameters();
     c2=neural_network.get_parameters();
    // cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@client2"<<endl;
     cout<<"client2_pararms:"<<client2_pararms<<endl;
   s2="";
    for(int i=0;i<43;i++)
    {

       client22_pararms(i)=client2_pararms(i);
       std::stringstream ss;
    
       ss << client2_pararms(i);
       std::string asString = ss.str();
       s2+=asString;
       s2+=';';
    }
  

    cout << "\nConfusion matrix:\n" << confusion << endl;

    
 
}
class MyApp2 : public Application 
{
public:
  void trainning(void);
};
void MyApp2::trainning(void)
{
       
   cout << "Client3 OpenNN. Iris Plant Example." << endl;

    srand(static_cast<unsigned>(time(nullptr)));

   
    DataSet data(data3);
   
    data.set_input_target_columns(input_indices,target_indices);


    const Index input_variables_number1 = data_set.get_input_variables_number();
    const Index target_variables_number1 = data_set.get_target_variables_number();
    

   // const Index input_variables_number = data.get_input_variables_number();
   // const Index target_variables_number = data.get_target_variables_number();

    // Neural network

    const Index hidden_neurons_number = 3;
    Tensor<Index, 1> architecture(3);
    architecture[0] = input_variables_number1;
    architecture[1] = 5;
    architecture[2] = target_variables_number1;
    
    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

   Tensor<type,1> avg_pararms(43),a;//平均之后的 

    if(epoch>0)
    {
      for(int i=0;i<43;i++)
      {
         avg_pararms(i)=client33_pararms(i)+(client1_pararms2(i)-client33_pararms(i))*0.5*0.001;
       //a=client3_pararms+(client1_pararms-client3_pararms)*c3.constant(0.5f)*c3.constant(0.1f);
      }  
       //neural_network.set_parameters(a);
       neural_network.set_parameters(avg_pararms);
    }
   

    // Training strategy

    TrainingStrategy training_strategy(&neural_network, &data);
    //TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
   // training_strategy.set_display_period(1);
    training_strategy.set_maximum_epochs_number(1);




    training_strategy.perform_training1();

    // Testing analysis

    //const TestingAnalysis testing_analysis(&neural_network, &data_set);
    const TestingAnalysis testing_analysis(&neural_network, &data);


    const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

     client3_pararms=neural_network.get_parameters();
     
     c3=neural_network.get_parameters();
     

    // cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%client3"<<endl;
     //cout<<client3_pararms<<endl;
     s3="";

    for(int i=0;i<43;i++)
    {

       client33_pararms(i)=client3_pararms(i);
       std::stringstream ss;
    
       ss << client3_pararms(i);
       std::string asString = ss.str();
       s3+=asString;
       s3+=';';
    }
  

    cout << "\nConfusion matrix:\n" << confusion << endl;

    
 
}


int main (int argc, char *argv[])
{
  data4=data_set.get_data();
  input_indices=data_set.get_input_columns_indices();
  target_indices=data_set.get_target_variables_indices();
 
  CommandLine cmd;
  cmd.Parse (argc, argv);
  
  Time::SetResolution (Time::NS);

  
  nodes.Create (3);
  Ptr<MyApp> app1 = CreateObject<MyApp> ();
  nodes.Get (0)->AddApplication (app1); 
  Ptr<MyApp1> app2 = CreateObject<MyApp1> ();
  nodes.Get (1)->AddApplication (app1); 
   Ptr<MyApp2> app3 = CreateObject<MyApp2> ();
  nodes.Get (2)->AddApplication (app2); 
 /////////////////////////////////////////////////
  NodeContainer n0=NodeContainer(nodes.Get(0),nodes.Get(1));
  NodeContainer n1=NodeContainer(nodes.Get(2),nodes.Get(1));
 
  
  ////////////////////////////////////////////////
  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  
  NetDeviceContainer d0 = pointToPoint.Install (n0);
  NetDeviceContainer d1 = pointToPoint.Install (n1);

  

  InternetStackHelper stack;
  stack.Install (nodes);
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  address.Assign(d0);
  address.SetBase ("10.1.2.0", "255.255.255.0");
  address.Assign(d1);

  

  Ipv4InterfaceContainer interfaces = address.Assign (d0);

  Ipv4InterfaceContainer interfaces2 = address.Assign (d1);


  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  nodes.Get(0)->lruCache= new LRUCache(150);
  nodes.Get(1)->lruCache= new LRUCache(150);
  nodes.Get(2)->lruCache= new LRUCache(150);



/******************node0***********************/
  ifstream inFile1("/home/allen/Data/iris_plant.csv",ios::in);
  if(!inFile1)
  {
    cout<<"打开文件失败！"<<endl;
    exit(1);
  }
 
  string line1;
  int j_0=0;
  while(getline(inFile1,line1))
  {
      string temp;
   //   cout<<line1<<endl;
      nodes.Get(0)->lruCache -> set(j_0, line1);
    
      temp=nodes.Get(0)->lruCache -> get(j_0);
     // cout<<temp<<endl;
      j_0++;
      float number_0[7];
      change(temp,number_0);
      for(int i=0;i<7;++i)
      {
        data1(tensor_size1,i)=number_0[i];
       }

     tensor_size1++;
   
  }
  node_size[0]=tensor_size1;
  inFile1.close();
 
 
/******************node1***********************/
  ifstream inFile2("/home/allen/Data/iris_plant.csv",ios::in);
  if(!inFile2)
  {
    cout<<"打开文件失败！"<<endl;
    exit(1);
  }
  int m=0;

  string line2;
  int lp=0;
  while(getline(inFile2,line2))
  {
      string temp;
      nodes.Get(1)->lruCache -> set(lp, line2);
      temp=nodes.Get(1)->lruCache -> get(lp);
      lp++;
      float number_1[7];
      change(temp,number_1);
      for(int i=0;i<7;++i)
      {
        data2(tensor_size2,i)=number_1[i];
      }

       tensor_size2++;
  }
  node_size[1]=tensor_size2;

  inFile2.close();

  /******************node2***********************/
  ifstream inFile3("/home/allen/Data/iris_plant.csv",ios::in);
  if(!inFile3)
  {
    cout<<"打开文件失败！"<<endl;
    exit(1);
  }
  int m1=0;

  string line3;
  int lp1=0;
 // cout<<"第三个节点"<<endl;
  while(getline(inFile3,line3))
  {
     string temp;
      nodes.Get(2)->lruCache -> set(lp1, line3);
      temp=nodes.Get(2)->lruCache -> get(lp1);
      lp1++;
      float number_2[7];
      change(temp,number_2);
      for(int i=0;i<7;++i)
      {
        data3(tensor_size3,i)=number_2[i];
      }

       tensor_size3++;
    
  }
  node_size[2]=tensor_size3;

  inFile3.close();



 /**************************设置发送端和接收端*********************************/

   /*********************node0 as receive*******************/
  TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");//用于为应用程序创建套接字的工厂类型
  Ptr<Socket> Recv_sock = Socket::CreateSocket(nodes.Get (0), tid);
  //配置应用程序客户端，interfaces.GetAddress(1)获取服务器节点的ip地址
  InetSocketAddress addr = InetSocketAddress(interfaces.GetAddress(0), 10000);
  Recv_sock->Bind(addr);
  /////设置回调函数
  Recv_sock->SetRecvCallback(MakeCallback(&RecvString1));

  

  /***************************node1 as receive**********************************/
  Ptr<Socket> Recv_sock3 = Socket::CreateSocket(nodes.Get (1), tid);
  //配置应用程序客户端，interfaces.GetAddress(1)获取服务器节点的ip地址
  InetSocketAddress addr3 = InetSocketAddress(interfaces.GetAddress(1), 10000);
  Recv_sock3->Bind(addr3);
  /////设置回调函数
  Recv_sock3->SetRecvCallback(MakeCallback(&RecvString2));

  Ptr<Socket> Recv_sock4 = Socket::CreateSocket(nodes.Get (1), tid);
  InetSocketAddress addr4 = InetSocketAddress(interfaces2.GetAddress(1), 10000);
  Recv_sock4->Bind(addr4);
  Recv_sock4->SetRecvCallback(MakeCallback(&RecvString3));

  /****************************************************************************/
  
   /***************************node2 as receiver**********************************/

  Ptr<Socket> Recv_sock6 = Socket::CreateSocket(nodes.Get (2), tid);
  InetSocketAddress addr6 = InetSocketAddress(interfaces2.GetAddress(0), 10000);
  Recv_sock6->Bind(addr6);
  Recv_sock6->SetRecvCallback(MakeCallback(&RecvString4));

   /***************************************************************************/


   /***********************************sender**************************************/

  Ptr<Socket> Send_sock1 = Socket::CreateSocket(nodes.Get (0), tid);
  InetSocketAddress RecvAddr = InetSocketAddress(interfaces.GetAddress(1), 10000);
  Send_sock1->Connect(RecvAddr);
  ///////////////////////////////////////
  Ptr<Socket> Send_sock2 = Socket::CreateSocket(nodes.Get (1), tid);
  InetSocketAddress RecvAddr2 = InetSocketAddress(interfaces.GetAddress(0), 10000);
  Send_sock2->Connect(RecvAddr2);
  ///////////////////////////////
  Ptr<Socket> Send_sock3 = Socket::CreateSocket(nodes.Get (1), tid);
  InetSocketAddress RecvAddr3 = InetSocketAddress(interfaces2.GetAddress(0), 10000);
  Send_sock3->Connect(RecvAddr3);

    ///////////////////////////////
  Ptr<Socket> Send_sock4 = Socket::CreateSocket(nodes.Get (2), tid);
  InetSocketAddress RecvAddr4 = InetSocketAddress(interfaces2.GetAddress(1), 10000);
  Send_sock4->Connect(RecvAddr4);

   ///////////////////////////////
  
  /*string s11="11.1;11;";
  string s22="33,3;4.5;";
  string s33="22.2;8.8;";*/
    

       // cout<<"node_1:"<<endl;
       /* for(int i=0;i<tensor_size1;i++)
        {
           string temp;
           int j=0;
           temp=nodes.Get(0)->lruCache -> get(i);
           cout<<temp<<endl;
           float number_2[7];
           change(temp,number_2);
           for(int i=0;i<7;++i)
           {
              data1(j,i)=number_2[i];
           }
           j++;
        }
  //cout<<"node_2"<<endl;

        for(int i=0;i<tensor_size2;i++)
        {
           string temp;
           int j=0;
           temp=nodes.Get(1)->lruCache -> get(i);
        // cout<<temp<<endl;
           float number_2[7];
           change(temp,number_2);
           for(int i=0;i<7;++i)
           {
              data2(j,i)=number_2[i];
           }
           j++;
           
        }


       for(int i=0;i<tensor_size3;i++)
       {
           string temp;
           int j=0;
           temp=nodes.Get(2)->lruCache -> get(i);
           //cout<<"node_3:"<<endl<<temp<<endl;
           float number_2[7];
           change(temp,number_2);
           for(int i=0;i<7;++i)
           {
              data3(j,i)=number_2[i];
           }
           j++;
           
       }*/


 for(epoch=0;epoch<2;epoch++)
 {
      cout<<"****************************第"<<epoch<<"次迭代************************"<<endl;

      //app1->SetStartTime (Seconds (0.1));
    //  app1->trainning();
     // app1->SetStopTime (Seconds (t+500));

      /*app2->SetStartTime (Seconds (t));
      app2->SetStopTime (Seconds (t+50));

      app3->SetStartTime (Seconds (t));
      app3->SetStopTime (Seconds (t+50));*/
    

      
   
      if(epoch>0)
      {
        //  cout<<"s1:"<<endl<<s1<<endl;

          Simulator::Schedule(Seconds(t),&send,Send_sock1,s1); //

         // Simulator::Run ();
      

          Simulator::Schedule(Seconds(t),&send,Send_sock2,s2); //fagizuobian
        //  Simulator::Run ();
  
          //cout<<"*******************"<<endl<<Recv_sock<<endl;

       
          Simulator::Schedule(Seconds(t),&send,Send_sock3,s2); //fageiyoubian
        //  Simulator::Run ();
          
   
          //cout<<"s3:"<<endl<<s3<<endl;
     
         Simulator::Schedule(Seconds(t),&send,Send_sock4,s3); //fageizuobian*/
         Simulator::Run ();


     }
       app1->trainning();
       app2->trainning();
       app3->trainning();

      /* app1->SetStartTime (Seconds (t));
       app1->SetStopTime (Seconds (t+100));

       app2->SetStartTime (Seconds (t));
        app2->SetStopTime (Seconds (t+20));
       app3->SetStartTime (Seconds (t));
        app3->SetStopTime (Seconds (t+20));*/
       //app->SetStopTime (Seconds (20.));
     
     //}
    
      
    
    
   }

  
 // Simulator::Stop(Seconds(1000));
  Simulator::Destroy ();

  
  return 0;
}


void 
send(Ptr<Socket> sock,string str)
{ 
  Ptr<Packet> p = Create<Packet>(str);//把str写入到包内
  sock->Send(p);
} 
void RecvString1(Ptr<Socket> sock)//回调函数
{


  
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  
  cout<<"**************************************************************************"<<endl<<buffer<<endl;
  int k=0;

/*  ofstream fout("/home/allen/Data/1.txt");
  fout<<buffer;
  fout<<endl;
  fout.close();*/
  

 //cout<<"RecvString1:"<<endl<<buffer<<endl;
    //float *num;
    //float number[43];

   for(int i=0;i<buffer.size();i++)
   {  
      if(buffer[i]!=';')
      { 

        temp=temp+buffer[i];
      }
      if(buffer[i]==';')
      {
        
          number[k]=stof(temp);
          k++;
          temp=""; 
     }
     if(i==buffer.size()-2)
     {
        number[k]=stof(temp);
     }

   }

  /********************把数据写入Tensor变量*****************/
 
   
 //ofstream fout("/home/allen/Data/1.txt");
  for(int i=0;i<43;++i)
  {
     client1_pararms0(i)=number[i];
    // fout<<number[i]<<";";


  }

        
    //fout.close();
 // return num;
 // cout<<"client1_pararms0"<< client1_pararms0<<endl;
   
 }
void
RecvString2(Ptr<Socket> sock)//回调函数
{
  
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  

//cout<<"RecvString2:"<<endl<<buffer<<endl;

  /*ofstream fout("/home/allen/Data/2.txt");
  fout<<buffer;
  fout<<endl;
  fout.close();*/

   int k=0;
  
  float number[43];

   for(int i=0;i<buffer.size();i++)
   {  
      if(buffer[i]!=';')
      { 

        temp=temp+buffer[i];
      }
      if(buffer[i]==';')
      {
        
          number[k]=stof(temp);
          k++;
          temp=""; 
     }
     if(i==buffer.size()-2)
     {
        number[k]=stof(temp);
     }

   }

  /********************把数据写入Tensor变量*****************/
 
   

  for(int i=0;i<43;++i)
  {
     client0_pararms1(i)=number[i];

  }

   
 }
void
RecvString3(Ptr<Socket> sock)//回调函数
{
  
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  

  // cout<<"RecvString3:"<<endl<<buffer<<endl;

 /* ofstream fout("/home/allen/Data/3.txt");
  fout<<buffer;
  fout<<endl;
  fout.close();*/

   int k=0;
  
   float number[43];

   for(int i=0;i<buffer.size();i++)
   {  
      if(buffer[i]!=';')
      { 

        temp=temp+buffer[i];
      }
      if(buffer[i]==';')
      {
        
          number[k]=stof(temp);
          k++;
          temp=""; 
     }
     if(i==buffer.size()-2)
     {
        number[k]=stof(temp);
     }

   }

  /********************把数据写入Tensor变量*****************/
 
   

  /*for(int i=0;i<43;++i)
  {
     client2_pararms1(i)=number[i];

  }
  flag3=1;*/
 }

void
RecvString4(Ptr<Socket> sock)//回调函数
{
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  
  
  // change(buffer,number);
  /*ofstream fout("/home/allen/Data/4.txt");
  fout<<buffer;
  fout<<endl;
  fout.close();*/

   int k=0;
  
   float number[43];

   for(int i=0;i<buffer.size();i++)
   {  
      if(buffer[i]!=';')
      { 

        temp=temp+buffer[i];
      }
      if(buffer[i]==';')
      {
        
          number[k]=stof(temp);
          k++;
          temp=""; 
     }
     if(i==buffer.size()-2)
     {
        number[k]=stof(temp);
     }

   }
  // cout<<"RecvString4:"<<endl<<buffer<<endl;

  /********************把数据写入Tensor变量*****************/
 
   

  for(int i=0;i<43;++i)
  {
     client1_pararms2(i)=number[i];

  }
 }

void change(string s,float number[])
{
   int k=0;
   int i;
   string temp;

  for(i=0;i<s.size();i++)
   {  
      if(s[i]!=';')
      { 

        temp=temp+s[i];
      }
      if(s[i]==';')
      {
        
          number[k]=stof(temp);
          k++;
          temp=""; 
     }
     if(i==s.length()-1)
     {
        number[k]=stof(temp);
     }

   }
}
