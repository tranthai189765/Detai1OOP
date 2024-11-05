#include<iostream>
using namespace std;
double a[100];
double aver[100];
int main()
{
    double sum = 0;
    for(int i = 1 ; i <= 10 ;i++)
    {
        cin>> a[i];
        sum += a[i];
    }
    cout<<"KQ :"<<sum/10;
    cout<<endl;
    double sumaver = 0;
    for(int i = 1 ; i <= 9 ;i++)
    {
        cout<<(a[i+1]/a[i]-1)*100<<"%"<<endl;
        aver[i] = (a[i+1]/a[i]-1)*100;
        sumaver += aver[i];
    }
    cout<<"KQ AVERAGE:"<<sumaver/9<<"%"<<endl;
    cout<<"FROM : "<<(a[10]/a[1]-1)*100<<"%"<<endl;
}
