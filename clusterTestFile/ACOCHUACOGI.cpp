#include <bits/stdc++.h>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
using namespace std;
struct Point{
double x;
double y;
};
int number_of_cove;
int requiresensors;
int kco;
vector<double> number_of_targets;
int k = 1;
int rs;
int num[300];
double nfac[300];
int number_targets;
Point targets[300];
Point steinerpoint[300];
bool issteiner[300];
double distance(Point a, Point b)
{
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
std::mt19937& getRandomEngine() {
    static std::random_device rd;
    static std::mt19937 g(rd());
    return g;
}

int getRandomNumber(int n) {
    if (n < 1) {
        throw std::invalid_argument("The value of n must be at least 1.");
    }
    std::uniform_int_distribution<int> distribution(1, n);
    return distribution(getRandomEngine());
}

void selectRandomNumbers(int n, int result[3]) {
    if (n < 3) {
        cout << "Error: The value of n must be at least 3." << endl;
        exit(1); // Exit if n is less than 3
    }

    int selectedNumbers[3] = {-1, -1, -1};
    int count = 0;

    while (count < 3) {
        int num = getRandomNumber(n);
        bool isDuplicate = false;
        for (int i = 0; i < count; ++i) {
            if (selectedNumbers[i] == num) {
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate) {
            selectedNumbers[count] = num;
            ++count;
        }
    }
    for (int i = 0; i < 3; ++i) {
        result[i] = selectedNumbers[i];
    }
}

double getRanreal() {
    std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(getRandomEngine());
}
Point third(Point m1,Point m2)
{
    Point m3;
    m3.x = (m1.x + m2.x)/2 - 0.8660254 * (m1.y - m2.y);
    m3.y = (m1.y + m2.y)/2 + 0.8660254 * (m1.x - m2.x);
    return m3;
}
Point ethird(Point m1,Point m2,Point m3)
{
    if(distance(third(m1,m2),m3)<distance(third(m2,m1),m3)) return(third(m2,m1));
    else return(third(m1,m2));
}
double goc(Point m1,Point m2,Point m3)
{
    return (pow(distance(m1,m2),2)+pow(distance(m1,m3),2)-pow(distance(m3,m2),2))/(2*distance(m1,m2)*distance(m1,m3));
}
Point Fermat(Point m1,Point m2,Point m3)
{
    Point F;
    if((m1.x==m2.x)&&(m1.y==m2.y)&&(m1.x==m3.x)&&(m1.y==m3.y)) return m1;
    else if((m1.x==m2.x)&&(m1.y==m2.y))
    {
        F.x=(m2.x+m3.x)/2;
        F.y=(m2.y+m3.y)/2;
        return F;
    }
     else if((m3.x==m2.x)&&(m3.y==m2.y))
    {
        F.x=(m2.x+m1.x)/2;
        F.y=(m2.y+m1.y)/2;
        return F;
    }
     else if((m1.x==m3.x)&&(m1.y==m3.y))
    {
        F.x=(m2.x+m3.x)/2;
        F.y=(m2.y+m3.y)/2;
        return F;
    }
    else
    {
    if(goc(m1,m2,m3)<(-0.5)||goc(m1,m2,m3)==(-0.5)) return m1;
    else if(goc(m2,m1,m3)<(-0.5)||goc(m2,m1,m3)==(-0.5)) return m2;
    else if(goc(m3,m1,m2)<(-0.5)||goc(m3,m1,m2)==(-0.5)) return m3;
    else{
    Point g3 = ethird(m1,m2,m3);
    Point g1 = ethird(m3,m2,m1);
    double a1= (g3.y-m3.y)/(g3.x-m3.x);
    double b1 = g3.y-g3.x*a1;
    double a2= (g1.y-m1.y)/(g1.x-m1.x);
    double b2 = g1.y-g1.x*a2;
    F.x=(b2-b1)/(a1-a2);
    F.y=a1*F.x+b1;
    return F;
    }
}
}
bool check(Point fermat, Point m1, Point m2, Point m3)
{
    if((fermat.x == m1.x)&&(fermat.y==m1.y)) return false;
    if((fermat.x == m2.x)&&(fermat.y==m2.y)) return false;
    if((fermat.x == m3.x)&&(fermat.y==m3.y)) return false;
    return true;
}
void addition(Point targets[],Point solution[],int &n,int &k1)
{
    //cout<<"UPDATE SOLUTION ADDITION: " <<endl;
    int result[3];
    selectRandomNumbers(n+k1,result);
    Point a[3];
    for(int i=0;i<3;i++)
    {
        if(result[i] <n ||result[i]==n)
        {
            a[i]=targets[result[i]];
         //   cout<<"CHON TARGET : "<<result[i]<<endl;
        }
        else {
                a[i]=solution[result[i]-n];
               // cout<<"CHON SOLUTION : "<<result[i]-n<<endl;
             }
    }
    Point f = Fermat(a[0],a[1],a[2]);
    if(check(f,a[0],a[1],a[2])==true)
    {
    k1++;
    solution[k1]= Fermat(a[0],a[1],a[2]);
   // cout<<"ADD SOLUTION : "<<solution[k1].x<< " "<<solution[k1].y<<endl;
    //cout<<"K1 = "<<k1<<endl;
    }
    else
    {
        return;
    }
}
void deletion(Point targets[],Point solution[],int &n, int &k1)
{
    //cout<<"UPDATE SOLUTION DELETION: "<<endl;
    int del= getRandomNumber(k1);
    //cout<<"XOA SOLUTION : "<<del<<endl;
    for(int i=del;i<=k1-1;i++)
    {
        solution[i].x=solution[i+1].x;
        solution[i].y=solution[i+1].y;
    }
    k1--;
}
void replacement(Point targets[],Point solution[],int &n,int &k1)
{
   // cout<<"UPDATEEEEEEEEE------------REPLACEMENT"<<endl;
    addition(targets,solution,n,k1);
    deletion(targets,solution,n,k1);
    //cout<<"END REPLACEMENT--------"<<endl;
}
Point neighbor1[300];
int kneigh;
void getneighbor1(Point targets[],Point solution[],int &n,int &k1)
{
 if(k1==0) addition(targets,solution,n,k1);
 else if (k1==(n-2))
    {
        int q = getRandomNumber(2);
        if(q==1) deletion(targets,solution,n,k1);
        else if (q==2) replacement(targets,solution,n,k1);
    }
else
    {
        int r = getRandomNumber(3);
        if(r==1) deletion(targets,solution,n,k1);
        else if (r==2) replacement(targets,solution,n,k1);
        else if (r==3) addition(targets,solution,n,k1);
    }
}
void copyy1(Point candidates[],Point tage[], int n)
{
    if(n==0) return;
    for(int i = 1 ;i<=n;i++)
    {
        candidates[i].x = tage[i].x;
        candidates[i].y = tage[i].y;
    }
}
void getneighbor2(Point targets[], Point solution[],int n, int k1)
{
    kneigh = 0;
    copyy1(neighbor1,solution,k1);
    kneigh = k1;
    getneighbor1(targets,neighbor1,n,kneigh);
}
void copyy(int candidates[],int tage[], int n)
{
    for(int i = 1 ;i<=n;i++)
    {
        candidates[i] = tage[i];
    }
}
void refresh(int solution[],int i, int n)
{
    for( int j = i; j<= n ;j++)
    {
        solution[j] = 1e9;
    }
}
int minKey(double key[], bool mstSet[], int V) {
     double min = DBL_MAX;int min_index;

    for (int v = 0; v < V; v++)
        if (!mstSet[v] && key[v] < min)
            min = key[v], min_index = v;

    return min_index;
}
int le =0;
Point leaf[300];
int degree[300]={0};
int parent[300];
double key[300];
bool mstSet[300];
double primMST(Point m[], double graph[300][300], int V) {
    double total = 0;
    for (int i = 0; i < V; i++) {
        key[i] = INT_MAX;
        mstSet[i] = false;
    }
    key[0] = 0;
    parent[0] = -1;
    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet, V);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
    for (int i = 1; i < V; i++) {
        if(graph[i][parent[i]]<(2*rs)) total =total+0;
        else total += ceil((static_cast<double>(graph[i][parent[i]]))/(2*rs))-1;
        degree[i]++;
        degree[parent[i]]++;
    }
    for (int i = 0; i < V; i++) {
            if(degree[i]==1)
            {
                leaf[le].x=m[i].x;
                leaf[le].y=m[i].y;
                le++;
            }
    }
    return total;
}
Point ta[300];
double graph[300][300];
double graph1[300][300];
double fitness1(int tage[],int n,int kco)
{
    le=0;
    int V=n;
    for(int i=0;i<n;i++)
    {
        ta[i].x=targets[tage[i+1]].x;
        ta[i].y=targets[tage[i+1]].y;
    }
    for (int i = 0; i < V; i++) {
        graph[i][i] = 0;
    }
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            graph[i][j] = distance(ta[i], ta[j]);
            graph[j][i] = graph[i][j];
        }
    }
    double t = primMST(ta,graph, V);
    for (int i = 0; i < le; i++) {
        for (int j = 0; j < le; j++) {
            if (i == j)
                graph1[i][j] = 0;
            else
                graph1[i][j] = distance(leaf[i], leaf[j]);
        }
    }
    double leafMstTotal = primMST(leaf, graph1, le);
    fill(degree, degree + 300, 0);
    le=0;
    double finall = ceil(static_cast<double>(kco)/2)*t + (kco/2)*leafMstTotal;
    return finall;
}
double fitness(int tage[],int n,int kco)
{
double t = fitness1(tage,n,kco);
int sum = 0;
for(int i = 1;i<=n;i++)
{
sum+= num[tage[i]];
if(issteiner[tage[i]]==true) t += ceil(static_cast<double>(kco)/2);
else if ((issteiner[tage[i]]==false) && (i>1 )) t+= kco;
}
if(t > requiresensors) return sum*sum*(static_cast<double>(requiresensors)/(t*t));
return sum;
}
double fitness2(Point targets[],Point solution[],int n,int k1,int kco)
{
    le=0;
    int V=(n+k1);
    for(int i=0;i<n;i++)
    {
        ta[i].x=targets[i+1].x;
        ta[i].y=targets[i+1].y;
    }
    for(int i=0;i<k1;i++)
    {
        ta[i+n].x=solution[i+1].x;
        ta[i+n].y=solution[i+1].y;
    }
    for (int i = 0; i < V; i++) {
        graph[i][i] = 0;
    }
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            graph[i][j] = distance(ta[i], ta[j]);
            graph[j][i] = graph[i][j];
        }
    }
    double t = primMST(ta,graph, V);
    for (int i = 0; i < le; i++) {
        for (int j = 0; j < le; j++) {
            if (i == j)
                graph1[i][j] = 0;
            else
                graph1[i][j] = distance(leaf[i], leaf[j]);
        }
    }
    double leafMstTotal = primMST(leaf, graph1, le);
    fill(degree, degree + 300, 0);
    le=0;
    double finall = ceil(static_cast<double>(kco)/2)*t + (kco/2)*leafMstTotal+kco*(n-1)+k1*ceil(static_cast<double>(kco)/2);
    //double finall = t  + leafMstTotal;
    return finall;
}
double soluongsensor(int tage[],int n,int kco)
{
double t = fitness1(tage,n,kco);
int sum = 0;
for(int i = 1;i<=n;i++)
{
sum+= num[tage[i]];
if(issteiner[tage[i]]==true) t += ceil(static_cast<double>(kco)/2);
else if ((issteiner[tage[i]]==false) && (i>1 )) t+= kco;
}
return t;
}
double initial_temperature1=100;
long long max_iterations1 = 1;
double cooling_rate1 =0.99;
int kmax = 0;
double initialvalue;
void SA1(Point targets[], Point solution[],int n,int k1,int kco, Point day[],int& numday)
{
   // cout<<"SENSORS BAN DAU : ----------"<<endl;
    Point best_solution1[300];
    Point current_solution1[300];
    // for(int i = 1 ;i<=n;i++)
    // {
       //  cout<<targets[i].x<<" "<<targets[i].y<<endl;
    // }
     double current_temperature1 = initial_temperature1;
     k1=0;
     double bestvalue1 = fitness2(targets,solution,n,k1,kco);
     initialvalue = bestvalue1;
    // cout<<"INITIAL VALUE : " <<bestvalue<<endl;
     for(int i = 0 ;i < max_iterations1;i++){
       // cout<<"LOOP : "<<i<<"----------------------------------------------"<<endl;
        double fitness_current1 = fitness2(targets,current_solution1,n,k1,kco);
        getneighbor2(targets,current_solution1,n,k1);
      //  cout<<"AFTER GETNEIGHBOR : -------------"<<endl;
        double new_value = fitness2(targets,neighbor1,n,kneigh,kco);
      //  cout<<"NEW_VALUE : "<< new_value<<endl;
        if(new_value < bestvalue1){
            copyy1(best_solution1,neighbor1,kneigh);
            bestvalue1 = new_value;
            kmax = kneigh;
        }
        double delta = -new_value + fitness_current1;
        if (delta > 0 || ((double)rand() / RAND_MAX) < exp(delta / current_temperature1)) {
          //  cout<<"acept ! "<<endl;
            copyy1(current_solution1, neighbor1, kneigh);
            k1 = kneigh;
        //   for(int i = 1;i<=k1;i++)
      //  {
        //    cout<<current_solution[i].x<<" "<<current_solution[i].y<<endl;
       // }
        }
        current_temperature1 *= cooling_rate1;
     }
     //cout<<"INITIAL VALUE = "<<initialvalue<<endl;
     //cout<<"BEST VALUE = "<<bestvalue1<<endl;
    // cout<<"SO DIEM STEINER = "<<kmax<<endl;
   //  cout<<"BEST SOLUTION : "<<endl;
     //for(int i = 1 ; i<= kmax;i++)
    // {
        // cout<<best_solution1[i].x<< " "<<best_solution1[i].y<<endl;
    // }
     numday = kmax;
     copyy1(day,best_solution1,numday);
     kmax=0;
}
void addingpoint(Point targets[], Point steiner[], int &n, int k1)
{
 //  cout<<"ADDING STEINER POINTS --- "<<endl;
  // cout<<"k1 = "<<k1<<endl;
  // cout<<"CHECK!"<<endl;
  // for(int i = 1;i<=k1;i++)
  // {
     //  cout<<steiner[i].x<<" "<<steiner[i].y<<endl;
  // }
  // cout<<"---------------------------"<<endl;
   for(int i = (n+1); i<=(n+k1);i++)
   {
       targets[i].x = steiner[i-n].x;
       targets[i].y = steiner[i-n].y;
       //cout<<"i = "<<i<<" --> "<<targets[i].x << " "<<targets[i].y<<endl;
   }
   n = n+k1;
  // cout<<"CAP NHAT n = "<<n<<endl;
}
int neighbor[300];
int U;
int V;
void getneighbor(int solution[], int candidates[], int numcan, int k1) {
     U = getRandomNumber(k1-1)+1;
     V = getRandomNumber(numcan);
    for (int i = 1; i <= k1; i++) {
        neighbor[i] = solution[i];
    }
    neighbor[U] = candidates[V];
}
int current_solution[300];
double initial_temperature=0.5;
int max_iterations = 1 ;
double cooling_rate=0.95;
void SA(int solution[],int candidates[], int k1, int numcan,double& subbestvalue) {
    if(numcan==0) return;
    copyy(current_solution,solution, k1);
    double current_temperature = initial_temperature;
    copyy(solution, current_solution, k1);
     subbestvalue = fitness(current_solution,k1,kco);
    for (int i = 0; i < max_iterations; i++) {
        double fitness_current = fitness(current_solution,k1,kco);
        U=0;V=0;
         getneighbor(current_solution,candidates,numcan,k1);
        double new_value = fitness(neighbor,k1,kco);
        if (new_value > subbestvalue) {
            copyy(solution, neighbor, k1);
            subbestvalue = new_value;
        }
        double delta = new_value - fitness_current;
        if (delta > 0 || ((double)rand() / RAND_MAX) < exp(delta / current_temperature)) {
            candidates[V] = current_solution[U];
            copyy(current_solution, neighbor, k1);
        }
        current_temperature *= cooling_rate;
    }
}
vector<Point> Set_Intersection(Point a, Point b){
    vector<Point> SI;
    if(distance(a, b) > 2 * rs || distance(a, b) == 0) return SI ;
    else{
        Point c;
        c.x = (a.x + b.x) / 2;
        c.y = (a.y + b.y) / 2;
        if(distance(a, b) == 2 * rs){
            SI.push_back(c);
            return SI;
        }
        else{
            long double A = a.x - b.x, B = a.y - b.y, C = (b.x - a.x) * c.x + (b.y - a.y) * c.y;
            Point X, Y;
            if(A == 0){
                X.y = -C / B;
                Y.y = -C / B;
                X.x = a.x - sqrt(rs * rs - (X.y-a.y) * (X.y-a.y));
                Y.x = a.x + sqrt(rs * rs - (X.y-a.y) * (X.y-a.y));
            }
            else if(B == 0){
                X.x = -C / A;
                Y.x = -C / A;
                X.x = a.y - sqrt(rs * rs - (X.x-a.x) * (X.x-a.x));
                Y.x = a.x + sqrt(rs * rs - (X.x-a.x) * (X.x-a.x));
            }
            else{
                long double a1 = 1 + (A * A) / (B * B);
                long double b1 = -2 * a.x + 2 * a.y * A / B + 2 * A * C / (B * B);
                long double c1 = a.x * a.x + a.y * a.y + pow(C / B, 2) -pow(rs, 2) + 2 * a.y * C / B;
                long double delta = fabs(b1 * b1 - 4 * a1 * c1);
                X.x = (-b1 + sqrt(delta)) / (2.0 * a1);
                Y.x = (-b1 - sqrt(delta)) / (2.0 * a1);
                X.y = -A / B * X.x - C / B;
                Y.y = -A / B * Y.x - C / B;

            }

            int u = k;
            if(u >= 2){
            SI.push_back(X);
            SI.push_back(Y);
            u -= 2;
            }
            else if(u == 1){
                SI.push_back(X);
                u--;
            }
            while(u > 0){
                SI.push_back(c);
                c.x = (c.x + X.x)/ 2;
                c.y = (c.y + X.y)/ 2;
                u--;
            }
            return SI;
        }
    }
}
vector<Point> CoverageS(vector<Point> targets, int n){
    vector<bool> mark(n, false);
    vector<Point> S;
    vector<Point> M;
    int h = n;

    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            vector<Point> p = Set_Intersection(targets[i], targets[j]);
            if(p.size() > 0){
                mark[i] = true;
                mark[j] = true;
                for(auto i : p)
                    M.push_back(i);

            }
        }
    }
    for(int i = 0; i < n; i++){
        if(mark[i] == false){
            number_of_cove++;
            number_of_targets.push_back(1);
            S.push_back(targets[i]);
            int u = k;
            if(u >= 2){
                Point t = targets[i];
                t.x += 1;
                t.y += 1;
                while(u - 1 > 0){
                    S.push_back(t);
                    t.x = (t.x + targets[i].x) / 2;
                    t.y = (t.y + targets[i].y) / 2;
                    u--;
                }
            }
            h--;
        }
    }
    int m = M.size();
    vector<bool> choosesensor(m, false);
    int v = 1;
    while(v > 0){
        int u = h;
        vector<bool> choosetarget(n, false);
        while(u > 0){
        int maxtargets = 0;
        Point choose;
        int index = 0;
        for(int i = 0; i < M.size(); i++){
            if(choosesensor[i] == false){
                int cnt = 0;
                for(int j = 0; j < n; j++){
                    if(mark[j] == true && choosetarget[j] == false){
                        if(distance(M[i], targets[j]) <= rs + 0.000005) cnt++;
                    }
                }
                if(cnt > maxtargets){
                    maxtargets = cnt;
                    choose = M[i];
                    index = i;
                }
            }
        }
            number_of_cove++;
            number_of_targets.push_back(maxtargets);
            S.push_back(choose);
            choosesensor[index] = true;
            u -= maxtargets;
            for(int i = 0; i < n; i++){
                if(distance(choose, targets[i]) <= rs + 0.000005) choosetarget[i] = true;
            }
        }
        v--;
    }
    return S;
}
double tvalue(Point m,int solution[],int n)
{
    double value = distance(m,targets[solution[n]]);
    if(value==0) return 0;
    return 1/value;
}
double alpha=3;
double beta=3;
double subprob(int ta, int solution[],int i,int m,int k)
{
    double fact = nfac[ta];
    double value = pow(fact, beta)*pow(tvalue(targets[ta], solution, m), alpha);
    return value;
}
double probability(int tage[], int solution[],int i,int n,int m,int k)
{
double sum = 0;
for(int i = 1 ;i<=n;i++)
{
sum += subprob(tage[i],solution,i,m,k);
}
return (subprob(tage[i],solution,i,m,k))/(sum);
}
double tmax=4;
double tmin=0.5;
double p=0.991;
int nbAnts = 200;
int tage[300];
int solution[300];
void removee(int j, int candidates[],int &n)
{
    for(int  i = j ;i <= (n-1);i++)
    {
        candidates[i] = candidates[i+1];
    }
    candidates[n] = 0;
    n--;
}
int bestsol[300][300];
Point steinerpointk[300][300];
int F[300];
int numsteiner[300];
int ni=0;
int ex[300];
Point steinercandi[300][300];
Point steinerp[300];
double subACO(int k,int loop,int n,int m,int kco)
{
    if(k>n) return 0;
    if(ex[k]!=0)
    {
       // cout<<"TINH ROI : VALUE = "<<F[k]<<endl;
        return F[k];
    }
  //  cout<<"K = "<<k<<"-----------------"<<endl;
    double bestvalue = -1e9;
      for(int i=1;i<=ni;i++){
        nfac[i]= num[i];
       }
      for(int i = (ni+1); i<= 180;i++)
      {
          nfac[i]=0.0000000000000005;
      }
    for(int u = 1 ;u <= loop;u++)
     {
       double subbestvalue = -1e9;
       double subminvalue = 1e9;
       double senbestvalue = 1e9;
       int solcan[300][300];
       double ketqua[300];
       double sum1=0;
       int vmax = 0;
       for(int v = 1;v <= nbAnts; v++)
        {
            double kq = 0;
            int k1=m+1;
            int numcan = n;
            double val = 0;
            refresh(solution,m+1,k+1);
            int ra = getRandomNumber(numcan);
            int candidates[300];
            copyy(candidates, tage,numcan);
            solution[k1] = candidates[ra];
            val = val + num[solution[k1]];
            removee(ra,candidates,numcan);
            while(k1 <= k)
            {
            double ranreal = getRanreal();
            double sum  =  0.0000001;
            int fi = 0;
            while (sum<= ranreal)
            {
                fi++;
                sum += probability(candidates,solution,fi,numcan,k1,k);
            }
            k1++;
            int thi = fi;
            solution[k1]= candidates[thi];
            val = val + num[solution[k1]];
            removee(thi,candidates,numcan);
            }
            int endk=k1;
            kq = val;
            SA(solution,candidates,k1,numcan,kq);
            double j = fitness1(solution,k+1,kco)+kco*k;
            copyy(solcan[v],solution,endk);
            ketqua[v] = kq;
            sum1+= ketqua[v];
            if((ketqua[v]>subbestvalue) || ((ketqua[v]==subbestvalue) &&(j < senbestvalue)))
            {
                subbestvalue= ketqua[v];
                vmax = v;
            }
        }
         if(subbestvalue > bestvalue)
           {
            bestvalue = subbestvalue;
            copyy(bestsol[k],solcan[vmax],k+1);
           }
        for(int v = 1;v <= nbAnts; v++)
         {
             for(int e =2; e <= (k+1);e++)
            {
                nfac[solcan[v][e]] = nfac[solcan[v][e]]+ ketqua[v]/(sum1*p);
            }
        }
         for(int e = 1; e<= n ;e++)
         {
            nfac[e] =  nfac[e]*p;
            if(nfac[e]< tmin) nfac[e]=tmin;
            else if(nfac[e]> tmax) nfac[e]=tmax;
         }
    }
     double t = soluongsensor(bestsol[k],k+1,kco);
     for(int i = 1 ; i<=(k+1);i++)
     {
        // cout<<targets[bestsol[k][i]].x<< " "<<targets[bestsol[k][i]].y<<endl;
         steinercandi[k][i].x = targets[bestsol[k][i]].x;
         steinercandi[k][i].y = targets[bestsol[k][i]].y;
     }
     int kk=0;
    // cout << "VALUE : "<<bestvalue<<endl;
    // cout << "-----------------------------"<<endl;
     //if(loop != 40) SA1(steinercandi[k],steinerp,k+1,0,kco,steinerpointk[k],kk);
     numsteiner[k]=kk;
    // cout<<"NUM STEINER[k] ="<< numsteiner[k]<<endl;
    // for(int i = 1;i<= numsteiner[k];i++)
    // {
     //    cout<<steinerpointk[k][i].x << " "<<steinerpointk[k][i].y<<endl;
    // }
//     SA1(steinercandik,steinerpo[k],k+1,kk,kco);
//     numsteiner[k]=kk;
     if(t> requiresensors) return 0;
     else
      {
    //    cout<<"NHUNG MA TINH ROI F[K] = "<<F[k]<<endl;
        if(F[k]>bestvalue) bestvalue = F[k];
        F[k]=bestvalue;
      //  cout<<"TINH LAI F[K] = "<<F[k]<<endl;
        ex[k]=1;
        return bestvalue;
      }
}
double bestbest = -1e9;
double kf = 0;
double subACO1(int k, int loop,int n,int m,int kco)
{
if(loop!=40)
{
//cout<<"KFI = "<<kf<<endl;
double check = subACO(k,loop,n,m,kco);
//cout<<"CHECK = "<<check<<endl;
//cout<<"BESTBEST = "<<bestbest<<endl;
//cout<<"KFI = "<<kf<<endl;
if((check < bestbest)&&(k<kf)) return 1;
else
{
double check2 = subACO(k-1,loop,n,m,kco);
if (check > bestbest)
{
    kf = k;
  //   cout<<"THAY DOI KFI1 = "<<kf<<endl;
    bestbest = check;
}
if (check2 > bestbest)
{
    kf = k-1;
  //   cout<<"THAY DOI KFI2 = "<<kf<<endl;
    bestbest = check2;
}
if(check==check2 && check ==0) return -1;
return check-check2;
}
}
else
{
//cout<<"KFI = "<<kf<<endl;
double check = subACO(k,loop,n,m,kco);
//cout<<"CHECK = "<<check<<endl;
//cout<<"BESTBEST = "<<bestbest<<endl;
//cout<<"KFI = "<<kf<<endl;
if((check < ((bestbest+1)))&&(k< ((kf+1)))) return 1;
else
{
double check2 = subACO(k-1,loop,n,m,kco);
//cout<<"CHECK LAN NUA ---"<<endl;
//cout<<"CHECK = "<<check<<endl;
//cout<<"BESTBEST = "<<bestbest<<endl;
//cout<<"CHECK2 = "<<check2<<endl;
if (check > bestbest)
{
    kf = k;
  //  cout<<"THAY DOI KFI1 = "<<kf<<endl;
    bestbest = check;
}
if (check2 > bestbest)
{
    kf = k-1;
    // cout<<"THAY DOI KFI2 = "<<kf<<endl;
    bestbest = check2;
}
if(check==check2 && check ==0) return -1;
return check-check2;
}
}
}
void loc(Point targets[], int& n)
{
    //cout<<"CHUAN BI LOC :--------------"<<endl;
   // for(int i = 1;i<=n;i++)
   // {
        //cout<<"i= "<<i<<" "<<targets[i].x<< " "<<targets[i].y<<endl;
  //  }
    Point loc[300];
    int numloc=0;
    copyy1(loc,targets,n);
    for(int i = 1 ;i<=n;i++)
    {
        bool isDuplicate = false;
        for (int j = 1; j <= numloc; j++) {
            if (loc[i].x == targets[j].x && loc[i].y == targets[j].y) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            numloc++;
            targets[numloc].x = loc[i].x;
            targets[numloc].y = loc[i].y;
        }
    }
  //  cout<<"LOC XONG ROI : --------------"<<endl;
   // for(int i = 1 ;i<=numloc;i++)
   // {
        //cout<<"i= "<<i<<" "<<targets[i].x<< " "<<targets[i].y<<endl;
   // }
    n = numloc;
    //cout<<"LOC XONG ROI THI N = "<<n<<endl;
    //cout<<"CHECK : "<<distance(targets[89],targets[88])<<endl;
}
void binsearch(int left, int right, int loop,int &n, int m,int kco)
{
   // cout << " N = "<<n<<endl;
    //cout<<"BESTBEST = "<<bestbest<<endl;
   // cout<<"KFI = "<<kf<<endl;
  //  cout<< " NUMBER OF TARGETS "<<number_targets<<endl;
     for(int i = 1;i <= n;i++)
      {
          numsteiner[i]=0;
        //  cout<<"TARGETS "<<i<<" "<<targets[i].x<<" "<<targets[i].y<<endl;
      }
      for(int i = 1 ;i<=300;i++) ex[i]=0;
    while(left <= (right+1))
    {
    int mid = (left+right)/2;
    double check = subACO1(mid,loop,n,m,kco);
    if(check>0) left = mid+2;
    else if(check==0||check<0) right = mid-2;
    }
   // cout<<"BESTBEST = "<<bestbest<<endl;
     cout<<bestbest/number_targets;
     int n2 = n;
     if(loop!=40)
     {
      for(int k = 1;k<=n2;k++)
     {
         if((bestbest-F[k])<3)
         {
           //  cout<<"k= "<<k<<" "<<"F[k] = "<<F[k]<<endl;
             SA1(steinercandi[k],steinerp,k+1,0,kco,steinerpointk[k],numsteiner[k]);
         }
     }
     }
     for(int i =1;i<=n2;i++)
      {
          addingpoint(targets,steinerpointk[i],n,numsteiner[i]);
      }
    //  cout<<"FINAL2 n = "<<n<<endl;
      for(int i = 1 ; i <=n;i++)
      {
        //  cout<<"i= "<<i<<" :  "<<targets[i].x << " "<<targets[i].y<<endl;
          if(i> n2)
          {
              issteiner[i]=true;
            //  cout<<"DAY LA DIEM STEINER THU "<<i<<endl;
          }
      }
      loc(targets,n);
}
int main()
{
    freopen("WSNACO.OUT", "w", stdout);
    for(int Test = 142  ;Test <= 400;Test++){
        string INP = to_string(Test);
        while(INP.size() < 4) INP = "0" + INP;
        INP = "t" + INP + ".txt";
        freopen(INP.c_str(), "r", stdin);
          cout<<"TEST "<<Test<<" : ";
           bestbest = -1e9;
           kf = 0;
        int p,q;
    cin>>p>>q;
    int m = 1;
     cin>>number_targets;
     cin>>requiresensors;
     cin>>kco;
     cin>>rs;
    for(int i=1;i<=m;i++)
    {
        solution[i]=0;
        double xc,yc;
        cin>>xc>>yc;
        targets[0].x=xc;
        targets[0].y=yc;
        tage[0]=0;
    }
    int n;
    vector<Point>nhapTargets(number_targets);
    for(int i = 0; i < number_targets; ++i)
        cin >> nhapTargets[i].x >> nhapTargets[i].y;
           number_of_targets.clear();
    vector<Point> cove = CoverageS(nhapTargets, number_targets);
    for(int i = 0; i < cove.size(); i++){
        targets[i + 1].x = cove[i].x;
        targets[i + 1].y = cove[i].y;
        tage[i+1]=i+1;
    }
    ni=0;
    n = cove.size();
    ni=n;
  //  cout<<"SO CUM = "<<n<<endl;
    solution[1]=0;
    for(int i=1;i<=m;i++) num[solution[i]]=0;

    for(int i=1;i<=ni;i++){
        num[i] = number_of_targets[i - 1];
        numsteiner[i]=0;
        nfac[i]= num[i];
        issteiner[i] = false;
    }
      for(int i = (ni+1); i<= 180;i++)
      {
          num[i]=0;
          tage[i]=i;
          nfac[i]=2;
          issteiner[i] = true;
      }
       for(int i = 1 ;i<=300;i++) F[i]=0;

      binsearch(1,n+1,110,n,m,kco);
    //  binsearch(1,n+1,40,n,m,kco);
      cout<<endl;
    }
}
