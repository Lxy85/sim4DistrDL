


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
#include <vector>
#include <iomanip>

using namespace OpenVGG;
using namespace ns3;
using namespace std;
using namespace OpenNN;

int row=70;
int col=70;
int number=600;
int channel=3;


int epoch=0;

//float number[43];

string s1="";
string s2="";
string s3="";

int tensor_size1=0;
int tensor_size2=0;
int tensor_size3=0;


float client1_to_server[162];
float client3_to_server[162];
float server_to_client1[162];
float server_to_client2[162];


bool count1=false;
bool receive1=false;
bool receive2=false;
int t=1;


static float image_new1[600][70][70][3];
static float image_new2[600][70][70][3];

NodeContainer nodes;
NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

void send(Ptr<Socket> sock,string str);

void send1(Ptr<Socket> sock,string str);

void acc(int a,int b);

void RecvString1(Ptr<Socket> sock);
void RecvString2(Ptr<Socket> sock);
void RecvString3(Ptr<Socket> sock);
void RecvString4(Ptr<Socket> sock);



void change(string s, float number[])
{
    int k = 0;
    int i;
    string temp="";
    for (i = 0; i < s.length(); i++)
    {
        if (s[i] != ',')
        {

            temp = temp + s[i];
        }
        if (s[i] == ',')
        {
            number[k] = stof(temp);
            k++;
            temp = "";
        }

    }
}


network net1(10, 70, 70, 3);
int *y_true1 = new int[10];
float ****image1 = get_mem<float>(10, 70, 70, 3);
load_data train_file1("/home/allen/Data/bird1.txt", 10, 70, 70, 3);

network net2(10, 70, 70, 3);
int *y_true2 = new int[10];
float ****image2 = get_mem<float>(10, 70, 70, 3);
load_data train_file2("/home/allen/Data/bird1.txt", 10, 70, 70, 3);




class MyApp1 : public Application 
{
public:
  void trainning(int batch,int data_size);
  void package(float****w,int i,int j,int m,int k);
 
};

void MyApp1::package(float****w,int row,int col,int c1,int c2)
{
   // cout<<"packet s1:"<<endl;
     s1="";
      for(int i=0;i<row;i++)
      {
        for(int j=0;j<col;j++)
        {
            for(int k=0;k<c1;k++)
            {
                for(int m=0;m<c2;m++)
                {
                     std::stringstream ss;
                     ss << w[i][j][k][m];
                     std::string asString = ss.str();
                     s1+=asString;
                     s1+=';';
                }
            }
        }
      }
     // cout<<s1<<endl;
}
void MyApp1::trainning(int batch,int data_size)
{
      // cout<<"********"<<endl;
       int num=data_size/batch;
       double loss_avg=0,acc_avg=0;
       int i;
       int k=0,r=1;
       double loss[60];
       double acc[60];

       int index=0;

       if(epoch>0)
       {
          for(int i=0;i<3;i++)
          {
            for(int j=0;j<3;j++)
            {
               for(int k=0;k<3;k++)
               {
                   for(int m=0;m<6;m++)
                  {
                     net1.conv2d1.w[i][j][k][m]=server_to_client1[index++];
                  }
               }
            }
           }
       }
        
       for(i=0;i<num;i++)
       {
         
         train_file1.next_batch(image1, y_true1,image_new1);
         //train_file1.next_batch(image1, y_true1);
         net1.forward(image1, y_true1);
         
         loss[i]=net1.get_loss();
         acc[i]=net1.get_accurate();
        // k++;
         cout<<endl<<i<<"th time is trainning"<<endl;
         cout<<"loss is "<<net1.get_loss()<<endl;
         cout<<"acc is "<<net1.get_accurate()<<endl;
     }
          


        for(int j=0;j<num;j++)
        {
              loss_avg+=loss[j];
              acc_avg+=acc[j];
         }
          
          loss_avg=loss_avg/num;
          acc_avg=acc_avg/num;
        
          cout<<"loss is:"<< loss_avg<<endl;
          cout<<"acc is:"<<acc_avg<<endl<<endl;

   

         net1.backward(image1, y_true1);
         net1.optim(0.001, 0.9, 0.99);
         //cout<<"##########&&&&&&#"<<net1.conv2d1.w[0][0][0][1]<<endl;

    //cout<<"###################"<<net1.conv2d1.w[0][0][0][0]<<endl;
      


     package(net1.conv2d1.w,3,3,3,6);

    /* s1="";
      for(int i=0;i<3;i++)
      {
        for(int j=0;j<3;j++)
        {
            for(int k=0;k<3;k++)
            {
                for(int m=0;m<6;m++)
                {

                    s1+=net1.conv2d1.w[i][j][k][m];
                    s1+=';';
                }
            }
        }
      }
      */
     // cout<<s1<<endl;
    
 
}
class MyApp2 : public Application 
{
public:
   
  /*void compute(float ****w,float ****w1,float ****w2,int batch, int m, int n, int c,int a1,int a2); 
  void compute_full(float **w,float **w1,float **w2,int m,int n,int a1,int a2);
  void compute_b(float *b,float *b1,float *b2,int m,int a1,int a2);*/
 // void package(float****w,int row,int col,int c1,int c2);  
  void calculate(void);

};

/************计算卷积层*************/
/*void MyApp2::compute(float ****w,float ****w1,float ****w2,int batch, int m, int n, int c,int a1,int a2)
{
    for(int i=0;i<batch;i++)
    {
        for(int j=0;j<m;j++)
        {
            for(int k=0;k<n;k++)
            {
                for(int p=0;p<c;p++)
                {
                    w[i][j][k][p] =  (w1[i][j][k][p]+ w2[i][j][k][p])*0.5;
                }
            }
        }
    }
} */
/************计算全连接层*************/
/*void MyApp2::compute_full(float **w,float **w1,float **w2,int m,int n,int a1,int a2)
{
    for(int i=0;i<m; i++)
    {
        for(int j=0;j<n;j++)
        {
            w[i][j] =(w1[i][j]+w2[i][j])*0.5;
        }
    }
}
void MyApp2::compute_b(float *b,float *b1,float *b2,int m,int a1,int a2)
{
    for(int i=0;i<m; i++)
    {
        
        b[i] =(b1[i]+b2[i])*0.5;
    }
}*/
/*void MyApp2::package(float****w,int row,int col,int c1,int c2)
{
     s2="";
      for(int i=0;i<row;i++)
      {
        for(int j=0;j<col;j++)
        {
            for(int k=0;k<c1;k++)
            {
                for(int m=0;m<c2;m++)
                {
                    s1+=w[i][j][k][m];
                    s1+=";";
                }
            }
        }
      }
}*/
void MyApp2::calculate(void)
{
    if(epoch>0)
    {
         count1=false;
         s2="";
      /* float w[3][3][3][6];
       for(int i=0;i<3;i++)
          {
            for(int j=0;j<3;j++)
            {
                for(int k=0;k<3;k++)
                {
                    for(int m=0;m<6;m++)
                    {
                        w[i][j][k][m]=(client1_to_server[i][j][k][m]+client3_to_server[i][j][k][m])*0.5;
                        s2+=w[i][j][k][m];
                        s2+=";";
                    }
                }
            }
          }*/
       float w[162];
       for(int i=0;i<162;i++)
        {
            
            w[i]=(client1_to_server[i]+client3_to_server[i])*0.5;
            std::stringstream ss;
            ss << w[i];
            std::string asString = ss.str();
            s2+=asString;
            s2+=';';
          
            
          }
          count1=true;

          cout<<"packet"<<"s2:"<<endl<<s2<<endl;
       
    }
    
}

class MyApp3 : public Application 
{
public:
  void trainning(int batch,int data_size);
  void computing(int i,int j,int m,int k);
  void package(float****w,int row,int col,int c1,int c2);
};
void MyApp3::package(float****w,int row,int col,int c1,int c2)
{
     s3="";

      for(int i=0;i<row;i++)
      {
        for(int j=0;j<col;j++)
        {
            for(int k=0;k<c1;k++)
            {
                for(int m=0;m<c2;m++)
                {
                     std::stringstream ss;
                     ss << w[i][j][k][m];
                     std::string asString = ss.str();
                     s3+=asString;
                     s3+=';';
                }
            }
        }
      }
     // cout<<"packet"<<"s3:"<<endl<<s3<<endl;
}
void MyApp3::trainning(int batch,int data_size)
{  
      cout<<"2"<<endl;
      int num=data_size/batch;
      float loss_avg=0,acc_avg=0;
      int i;
      float *loss = new float[num];
      float *acc = new float[num];

      int index=0;

       if(epoch>0)
       {
          for(int i=0;i<3;i++)
          {
            for(int j=0;j<3;j++)
            {
               for(int k=0;k<3;k++)
               {
                   for(int m=0;m<6;m++)
                  {
                     net2.conv2d1.w[i][j][k][m]=server_to_client2[index++];
                  }
               }
            }
           }
       }


      for(i=0;i<num;i++)
      {
         train_file2.next_batch2(image2, y_true2,image_new2);
        // train_file2.next_batch2(image2, y_true2);
         net2.forward(image2, y_true2);
         loss[i]=net2.get_loss();
         acc[i]=net2.get_accurate();
         net2.backward(image2, y_true2);
         net2.optim(0.001, 0.9, 0.99);
      }

      

      for(int j=0;j<i;j++)
      {
        loss_avg+=loss[j];
        acc_avg+=acc[j];
      }
      
      loss_avg=loss_avg/num;
      acc_avg=acc_avg/num;
    //  cout<<"client2"<<endl;
      cout<<"loss is:"<< loss_avg<<endl;
      cout<<"acc is:"<<acc_avg<<endl<<endl;

      package(net2.conv2d1.w,3,3,3,6);

    /*  s3="";
      for(int i=0;i<3;i++)
      {
        for(int j=0;j<3;j++)
        {
            for(int k=0;k<3;k++)
            {
                for(int m=0;m<6;m++)
                {
                    s3+=net2.conv2d1.w[i][j][k][m];
                    s3+=';';
                }
            }
        }
      }
      cout<<s3<<endl;*/
}


int main (int argc, char *argv[])
{
  
  CommandLine cmd;
  cmd.Parse (argc, argv);
  
  Time::SetResolution (Time::NS);

  
  nodes.Create (3);
  Ptr<MyApp1> app1 = CreateObject<MyApp1> ();
  nodes.Get (0)->AddApplication (app1); 
  Ptr<MyApp2> app2 = CreateObject<MyApp2> ();
  nodes.Get (1)->AddApplication (app2); 
   Ptr<MyApp3> app3 = CreateObject<MyApp3> ();
  nodes.Get (2)->AddApplication (app3); 
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

  nodes.Get(0)->lruCache= new LRUCache(660);
  nodes.Get(2)->lruCache= new LRUCache(660);


 int num_data=row*col*channel;
/******************node0***********************/
  ifstream inFile1("/home/allen/Data/bird1.csv",ios::in);
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
      tensor_size1++;
      float number_0[num_data];
      int num_0 = 0;
      int row1 = 0;
      int col1 = 0;
      int c = 0;

      change(temp,number_0);

      for (int i = 0; i < num_data; i++)
      {
            if (c == channel)
            {
                col1++;
                c = 0;
            }
            if (col1 == col)
            {
                row1++;
                col1 = 0;
            }
            if (row1 == row)
            {
                num_0++;
                row1 = 0;
            }
            image_new1[num_0][row1][col1][c]=number_0[i] ;
            c++;
      }
    
    }

  inFile1.close();
 
 
  /******************server***********************/
  
  /******************node2***********************/
  ifstream inFile3("/home/allen/Data/bird2.csv",ios::in);
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
     
   
      tensor_size2++;
      float number_0[num_data];
      int num_0 = 0;
      int row1 = 0;
      int col1 = 0;
      int c = 0;

      change(temp,number_0);

      for (int i = 0; i < num_data; i++)
      {
            if (c == channel)
            {
                col1++;
                c = 0;
            }
            if (col1 == col)
            {
                row1++;
                col1 = 0;
            }
            if (row1 == row)
            {
                num_0++;
                row1 = 0;
            }
            image_new2[num_0][row1][col1][c]=number_0[i] ;
            c++;
      }
  }

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



 /* for(int i=0;i<1;i++)
  {
      app1->trainning(10,10);
      app3->trainning(10,10);
      //app2->calculate();
  }*/

 for(epoch=0;epoch<40;epoch++)
 //for(int e=0;e<10;e++)
 {
      cout<<"****************************第"<<epoch<<"次迭代************************"<<endl;
   
      if(epoch>0)
      {
          Simulator::Schedule(Seconds(t),&send,Send_sock1,s1); 
     
          Simulator::Schedule(Seconds(t),&send,Send_sock4,s3); 

          t=t+20;
         
          Simulator::Schedule(Seconds(t),&send1,Send_sock2,s2); 
        
          Simulator::Schedule(Seconds(t),&send1,Send_sock3,s2); 
         

          Simulator::Run ();
            
         
      }
      app1->trainning(10,600);
      app3->trainning(10,600);
     // app2->calculate();
     
     /* cout<<"s1:"<<endl<<s1<<endl;
      cout<<"s2:"<<endl<<s2<<endl;
      cout<<"s3:"<<endl<<s3<<endl;*/
    
   }

  
Simulator::Destroy ();
 
  return 0;
}

void acc(int a,int b)
{ 
  /*Ptr<Packet> p = Create<Packet>(str);//把str写入到包内
  sock->Send(p);*/
} 


void 
send(Ptr<Socket> sock,string str)
{ 
  Ptr<Packet> p = Create<Packet>(str);//把str写入到包内
  sock->Send(p);
} 

void 
send1(Ptr<Socket> sock,string str)
{ 
  float w[162];
  s2="";
  for(int i=0;i<162;i++)
  {
            
     w[i]=(client1_to_server[i]+client3_to_server[i])*0.5;
     std::stringstream ss;
     ss << w[i];
     std::string asString = ss.str();
     s2+=asString;
     s2+=';';
          
            
  }
   Ptr<Packet> p = Create<Packet>(s2);//把str写入到包内
   sock->Send(p);
} 

void RecvString1(Ptr<Socket> sock)//回调函数
{
  // cout<<"*"<<endl;
    receive1=false;
   float number[162];
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  
  cout<<"server to client1"<<endl<<buffer<<endl;
  int k=0;

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

  /********************把数据写入变量*****************/
 
   
  for(int i=0;i<162;++i)
  {
     server_to_client1[i]=number[i];

  }
   receive1=true;
   
 }
void
RecvString2(Ptr<Socket> sock)//回调函数
{
  // cout<<"**"<<endl;
   receive2=false;
   string temp;
   Address from;
   Ptr<Packet> packet = sock->RecvFrom (from);
    
   packet->RemoveAllPacketTags ();
   packet->RemoveAllByteTags ();
   InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  
   std::string buffer;
   buffer.resize (packet->GetSize ());
   packet->CopyData (reinterpret_cast<uint8_t*> (&buffer[0]), buffer.size ());
  
   
   int k=0;
   float number[162];
  cout<<"client1 to server"<<endl<<buffer<<endl;
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
 
   

  for(int i=0;i<162;++i)
  {
     client1_to_server[i]=number[i];

  }
   receive2=true;

   
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
  
   cout<<"client3 to server"<<endl<<buffer<<endl;

   int k=0;
  
    float number[162];
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
 
   

  for(int i=0;i<162;++i)
  {
    client3_to_server[i]=number[i];

  }
   
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
  

    cout<<"server to client2"<<endl<<buffer<<endl;

   int k=0;
  
  float number[162];

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
 
   

  for(int i=0;i<162;++i)
  {
     server_to_client2[i]=number[i];

  }
 }

