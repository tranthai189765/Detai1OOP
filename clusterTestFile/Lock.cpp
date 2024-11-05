#include <bits/stdc++.h>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
using namespace std;
double a[201];
double b[201];
double kqa[201];
int main()
{
for (int i = 1;i<=100;i++)
{
while(true)
{
    string t;
    cin>>t;
    if(t=="TEST") break;
}
int k;
cin>>k;
char t;
cin>>t;
cin>> a[i]>>b[i];
cout<<b[i]<<endl;
}
for (int i = 0 ;i<=9;i++)
{
  double sum = 0;
  for (int j = 10*i+1; j<= 10*i+10;j++)
  {
      sum += b[j];
  }
  kqa[i+1] = sum/10;
}
for (int i = 1 ; i<= 10;i++)
{
    cout<<kqa[i]<<","<<endl;
}
}
