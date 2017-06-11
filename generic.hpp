#ifndef GENERIC_HPP
#define GENERIC_HPP

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <list>
#include <set>
#include <memory>
#include <stack>
#include <queue>
#include <algorithm>
#include <functional>
#include <math.h>
#include <time.h>

using namespace std;

/*
   Binary heap implementation
*/

template<class type>
struct MinHeapOpt
{
    static bool Compare(const type &dat1, const type &dat2)
    {
        return dat1 <= dat2;
    }
};

template<class type>
struct MaxHeapOpt
{
    static bool Compare(const type &dat1, const type &dat2)
    {
        return dat1 >= dat2;
    }
};

template<class DataType, class HeapType = MinHeapOpt<DataType>>
class BasicBinaryHeap
{
public:
    BasicBinaryHeap(size_t max_size = INT_MAX)
        :SizeMax_(max_size),
        DataSize_(0)
    {
        // arr[0] is reserved, start from 1
        Data_.push_back(DataType());
    }

    virtual ~BasicBinaryHeap() = default;

    void Push(const DataType &data)
    {
        if (DataSize_ < SizeMax_)
        {
            DataSize_++;
            Data_.push_back(data);
        }
        else Data_[DataSize_] = data; // overwirte the latest one
        Swim(DataSize_);
    }

    bool Empty() const
    {
        return DataSize_ == 0;
    }

    DataType Top() const
    {
        return Data_.size() > 1 ? Data_[1] : DataType();
    }

    void Pop()
    {
        // swap top and last
        swap(Data_[DataSize_], Data_[1]);
        Data_.pop_back();
        DataSize_--;
        Sink(1);
    }

private:
    bool Compare(int n1, int n2)
    {
        return HeapType::Compare(Data_[n1], Data_[n2]);
    }

    void Swim(int index)
    {
        // compare index and its parent index/2
        int parent = index / 2;
        while (index>1 && Compare(index, parent))
        {
            swap(Data_[index], Data_[parent]);
            index = parent;
            parent /= 2;
        }
    }

    void Sink(int index)
    {
        int child = index*2;
        while (child <= DataSize_)
        {
            // choose between left and right
            if (child < DataSize_ && Compare(child + 1, child)) child++;

            if (Compare(child, index))  swap(Data_[child], Data_[index]);
            else break;
            index = child;
            child *= 2;
        }
    }

private:
    size_t SizeMax_;
    size_t DataSize_;
    vector<DataType> Data_;
};

/*
Sort Summary
*/
template<class type>
class SortAlgo
{
public:
    SortAlgo() = default;
    virtual ~SortAlgo() {}

    virtual void Sort(vector<type> &arr) = 0;
    
    void Shuffle(vector<type> &arr)
    {
        //max of rand is defined by RAND_MAX 0x7fff
        
        // set random seed 
        srand((int)time(NULL));
        int N = arr.size();
        for (int i = 0; i < N; i++)
        {
            int r = rand()%N;
            swap(arr[i],arr[r]);
        }
    }
};

//  Heapsort
template<class type>
class HeapSort : public SortAlgo<type>
{
public:
    /* can be optimized as this
    void sort(Comparable[] a)
    {
    int N = a.length;
    for (int k = N/2; k >= 1; k--) sink(a, k, N);

    while (N > 1)
    {
        exch(a, 1, N--);
        sink(a, 1, N);
    }
    }
    */
    virtual void Sort(vector<type> &arr)
    {
        BasicBinaryHeap<type> sortor(arr.size());
        for (auto &dat : arr) sortor.Push(dat);

        for (int i = 0; i < arr.size(); i++)
        {
            arr[i] = sortor.Top();
            sortor.Pop();
        }
    }
};

// shell sort
template<class type>
class ShellSort : public SortAlgo<type>
{
public:
    // N^(6/5) ~ NlogN, unstable
    virtual void Sort(vector<type> &arr)
    {
        int N = arr.size();
        int factor = 1;
        // 1,4,13 
        while (factor < N / 3) factor = factor * 3 + 1; 

        while (factor > 0)
        {
            for (int i = factor; i < N; i++)
            {
                for (int j = i; j >= factor && 
                    arr[j] < arr[j-factor]; j -= factor) swap(arr[j],arr[j-factor]);
            }
            factor /= 3;
        }
    }
};

// quick sort
template<class type>
class QuickSort : public SortAlgo<type>
{
public:
    virtual void Sort(vector<type> &arr)
    {
        Shuffle(arr);
        qsort(arr,0,arr.size()-1);
    }

private:
    virtual void qsort(vector<type> &arr,int start ,int end)
    {
        if (start >= end) return;

        int mid = partition(arr,start,end);
        qsort(arr, start, mid-1);
        qsort(arr, mid+1, end);
    }

    int partition(vector<type> &arr, int start, int end)
    {
        int lo = start;
        int hi = end + 1;

        int split = arr[start]; 
        
        while (true)
        {
            // from start+1 to end, find candidator
            while (arr[++lo] < split) if (lo == end) break;
            // from end to lo, find candidator
            while (arr[--hi] > split) if (hi == start) break;

            if (lo < hi) swap(arr[lo], arr[hi]);
            else break;
        }
        swap(arr[start], arr[hi]);
        return hi;
    }
};

template<class type>
class QuickSortThreeWay : public QuickSort<type>
{
private:
    void qsort(vector<type> &arr, int start, int end)
    {
        if (start >= end) return;

        int hi = end;
        int lo = start;
        int index = lo + 1;
        int split = arr[lo];
        // split into 3 part: samller, equal, bigger 
        while (index <= hi)
        {
            if (arr[index] < split) swap(arr[lo++],arr[index++]);
            else if (arr[index] > split) swap(arr[hi--], arr[index]);
            else index++;
        }
        // arr[start..lo-1] <  split
        qsort(arr,start,lo-1);
        // arr[hi+1..end] >  split
        qsort(arr,hi+1,end);
    }
};

// merge sort
template<class type>
class MergeSort : public SortAlgo<type>
{
public:
    virtual void Sort(vector<type> &arr)
    {
        Aux_.resize(arr.size(),0);
        msort(arr,0,arr.size()-1);
    }
private:
    // top to down sort
    void msort(vector<type> &arr, int low, int high)
    {
        if (low >= high) return;

        int mid = low + (high - low) / 2;
        msort(arr, low, mid);
        msort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }

    // down to top sort
    void msort(vector<int> &arr)
    {
        int N = arr.size();
        // sub_size: subarray size
        for (int sub_size = 1; sub_size < N; sub_size *= 2)
        {
            // sub_id: subarray index
            for (int sub_id = 0; sub_id < N - sub_size; sub_id += sub_size*2)
                merge(arr, sub_id, sub_id + sub_size - 1, min(sub_id + sub_size*2 -1,N-1));
        }
    }

    // merge two sorted arry[low..mid,mid+1..high] into one
    void merge(vector<type> &arr, int low,int mid, int high)
    {
        // copy to array aux 
        for (int i = low; i <= high; i++) Aux_[i] = arr[i];
        
        // takes from tow arrys a1,a2
        int a1 = low, a2 = mid + 1;
        for (int i = low; i <= high; i++)
        {
            if (a1 > mid) arr[i] = Aux_[a2++]; // a1 is done
            else if (a2 > high) arr[i] = Aux_[a1++]; // a2 is done
            else if (Aux_[a1] < Aux_[a2]) arr[i] = Aux_[a1++]; // take a1
            else arr[i] = Aux_[a2++]; // take a2
        }
    }

private:
    vector<type> Aux_;
};

// merge sort in place
template<class type>
class MergeSortInplace : public SortAlgo<type>
{
public:
    virtual void Sort(vector<type> &arr)
    {
        msort(arr, 0, arr.size() - 1);
    }
private:
    void msort(vector<type> &arr, int low, int high)
    {
        if (low >= high) return;

        int mid = low + (high - low) / 2;
        msort(arr,low,mid);
        msort(arr,mid+1,high);
        merge(arr,low,mid,high);
    }

    void merge(vector<type> &arr, int low, int mid, int high)
    {
        int i = low, j = mid + 1;
        while (i < j && j <= high)
        {
            // i++ until arr[i] > arr[j]
            while (i < j && arr[i] <= arr[j]) i++;
            int index = j;
            // j++ until arr[j] > arr[i]
            while (j <= high && arr[j] < arr[i]) j++;
            // shift arr[i..index) [index..j)
            shift(arr,i,index-1,j-1);
            // move i to 
            i += j - index;
        }
    }
    // arr[start..split..end] -> arr[split..end..start]
    void shift(vector<type> &arr,int start, int split,int end)
    {
        flip(arr,start, split);
        flip(arr,split+1, end);
        flip(arr,start, end);
    }

    void flip(vector<type> &arr, int begin, int end)
    {
        while (begin < end) swap(arr[begin++], arr[end--]);
    }
};

// string LSD(least significant digit) sort
class StringLSDSort: public SortAlgo<string>
{
public:
    virtual void Sort(vector<string> &StrArr)
    {
        if (StrArr.size() == 0) return;

        int Radix = 256;

        int str_len = StrArr[0].size();
        int arr_len = StrArr.size();
        // from right to left LSD
        for (int d = str_len - 1; d >= 0; d--)
        {
            vector<int> count(Radix + 1, 0);
            // compute frequency of each character, count[r+1!!!]  
            for (int i = 0; i < arr_len; i++)
                count[StrArr[i].at(d) + 1]++;

            // change counts to indices
            for (int i = 0; i < Radix; i++)
                count[i + 1] += count[i];

            // distribute it to temp array
            // r = count[StrArr[i].at(d)]++ 
            // '++' here is to handle same character, move r to next position in aux
            vector<string> aux(arr_len, "");
            for (int i = 0; i < arr_len; i++)
                aux[count[StrArr[i].at(d)]++] = StrArr[i];

            // save back to string array
            for (int i = 0; i < arr_len; i++)
                StrArr[i] = aux[i];
        }
    }
};

/*
Union-Find
*/
class UnionFind
{
public:
    UnionFind() = default;
    ~UnionFind() = default;

    UnionFind(size_t size)
    {
        ResetUF(size);
    }

    void ResetUF(size_t size)
    {
        Member_.resize(size, 0);
        Rank_.resize(size, 0);
        for (int i = 0; i < size; i++)
        {
            Member_[i] = i;
            Rank_[i] = 1;
        }
        Count_ = size;
    }

    bool Connected(int id1, int id2)
    {
        return Find(id1) == Find(id2);
    }

    // union with rank
    void UnionWithRank(int id1,int id2)
    {
        int boss1 = Find(id1);
        int boss2 = Find(id2);

        if (boss1 == boss2) return;

        if (Rank_[boss1] < Rank_[boss2])
        {
            Member_[id1] = boss2;
            Rank_[boss2] += Rank_[boss1];
        }
        else
        {
            Member_[id2] = boss1;
            Rank_[boss1] += Rank_[boss2];
        }
        Count_--;
    }

    void Union(int id1, int id2)
    {
        int boss1 = Find(id1);
        int boss2 = Find(id2);

        if (boss1 == boss2) return;

        Member_[boss1] = boss2;
        Count_--;
    }

    // find with path compress
    /*
    int find(int id)
        if(id != arr[id]) arr[id] = find(arr[id]);
        return arr[id];
    */
    int Find(int id)
    {
        // find the boss of this union
        int boss = id;
        while (boss != Member_[boss]) boss = Member_[boss];

        int curr = id;
        int parent;
        // refresh the parent to boss
        while (curr != boss) 
        {
            parent = Member_[curr];
            Member_[curr] = boss;
            curr = parent;
        }
        //Rank_[boss] = 2; // path reset to 2
        return boss;
    }

private:
    size_t Count_;
    vector<int> Member_;
    vector<int> Rank_; // optional, prefere path compress
};

struct ComparerMax
{
    static int Value(int a, int b) { return max(a, b); }
};

struct ComparerMin
{
    static int Value(int a, int b) { return min(a, b); }
};
template<class Comparer = ComparerMin>
class MyRMQst
{
public:
    MyRMQst(vector<int> &nums)
    {
        SetData(nums);
    }
    MyRMQst() = default;
    ~MyRMQst() = default;

    // not correct
    void SetData(vector<int> &nums)
    {
        // F[i, j] = min(F[i, j-1], F[i+2^(j-1), j-1])
        int range = int(log(nums.size()) / log(2))+1;
        int num_size = nums.size();
        vector<int> temp(range, 0);
        Data_.clear();
        Data_.resize(num_size, temp);

        for (int i = 0; i < num_size; i++) Data_[i][0] = nums[i];

        for (int j = 1; j < range; j++)
        {
            for (int i = 0; i < num_size; i++)
            {
                int k = i + (1 << (j- 1));
                if (k < num_size)
                    Data_[i][j] = Comparer::Value(Data_[i][j - 1], Data_[k][j - 1]);
            }
        }
    }

    int MostValue(int i, int j)
    {
        if (i > j) swap(i, j);

        // k=log2(j-i+1) 
        int k = int(log(j - i + 1) / log(2));
        // RMQ(i,j) = min(F[i,k], F[j-2^k+1,k])
        return Comparer::Value(Data_[i][k], Data_[j + 1 - (1 << k)][k]);
    }

private:
    vector<vector<int>> Data_;
};

/*
Suffix Array Implementation
*/
class SuffixArray
{
public:
    SuffixArray(const string &str)
    {
        SetString(str);
    }
    SuffixArray() = default;
    virtual ~SuffixArray() {}

    virtual void SetupSuffixArray(const string &str)
    {
        SetString(str);
    }

    int Rank(const string &pat)
    {
        int lo = 0;
        int hi = SA_.size() - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            const char *suffix = SuffixStr(mid);
            int res = strcmp(pat.c_str(),suffix);
            if ( res < 0 )  hi = mid - 1;
            else if (res > 0) lo = mid + 1;
            else return mid;
        }

        return lo;
    }

    int Substring(const string &pat)
    {
        int lo = 0;
        int hi = SA_.size() - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            string suffix(SuffixStr(mid));
            // search for prefix matched string in suffix
            int res = suffix.compare(0, pat.size(), pat);
            if (res>0)  hi = mid - 1;
            else if (res<0) lo = mid + 1;
            else return SA_[mid];
        }

        return -1;
    }

    // Longest Common Substring
    string LCS(int split)
    {
        int max = 0;
        int index = 0;
        for (int i = 1; i < Height_.size(); i++)
        {
            if (Height_[i] > max &&
                (SA_[i - 1] - split)*(SA_[i] - split) < 0)
            {
                max = Height_[i];
                index = i;

            }
        }
        return string(SuffixStr(index)).substr(0, Height_[index]);
    }

    // Longest Palindromic Substring
    /*
    S="aabaaaab",  SS'="aabaaab$baaabaa",str(ss)=len , search from 1 to len(s)
    find the LCP(i and len-i) consider odd and even
    */
    string LPS(int split)
    {
        int max = 0;
        int index = 0;
        int str_size = RawString_.size();
        for (int i = 1; i < str_size / 2; i++)
        {
            // odd i is the center of palindromic string LCP(i,len-i-1) 
            // xx|i.......|(len-i-1)xxx
            int lcp = LCP(i, str_size - i - 1);
            if (lcp * 2 - 1 > max)
            {
                max = lcp * 2 - 1;
                index = i - lcp + 1;
            }
            // even xx|i.......|(len-i)xx
            lcp = LCP(i, str_size - i);
            if (lcp * 2 > max)
            {
                max = lcp * 2;
                index = i - lcp;
            }
        }
        return RawString_.substr(index, max);
    }

private:
    void SetString(const string &str)
    {
        RawString_ = str;
        SA_.clear();
        for (int i = 0; i < str.size(); i++)
        {
            SA_.push_back(i);
        }
        sort(SA_.begin(), SA_.end(), [&](int offset1, int offset2)
        { return strcmp(RawString_.c_str() + offset1, RawString_.c_str() + offset2) <= 0; });

        // calculate rank, the raw index in SA
        Rank_.resize(SA_.size(), 0);
        for (int i = 0; i < SA_.size(); i++)
        {
            Rank_[SA_[i]] = i;
        }
        // calculate height array height[i] = LCP(i,i-1)
        Height_.resize(SA_.size(), 0);
        for (int i = 1; i < SA_.size(); i++)
        {
            Height_[i] = TwoStringLCP(SuffixStr(i), SuffixStr(i - 1));
        }
    }

    const char* SuffixStr(int sa_index)
    {
        return RawString_.c_str()+SA_[sa_index];
    }

    // lcp(RawString[index1],RawString[index2])
    virtual int LCP(int index1, int index2)
    {
        int first = Rank_[index1];
        int second = Rank_[index2];
        if (first > second) swap(first, second);

        int min = INT_MAX;
        // from first+1 ~ second
        for (int i = first + 1; i <= second; i++)
        {
            if (min > Height_[i]) min = Height_[i];
        }
        return min;
    }

    int TwoStringLCP(const char *str1, const char *str2)
    {
        int len = min(strlen(str1), strlen(str2));
        for (int i = 0; i < len; i++)
        {
            if (str1[i] != str2[i]) return i;
        }
        return  len;
    }


protected:
    vector<int> SA_;
    vector<int> Rank_;
    vector<int> Height_;
    string RawString_;
};

class SuffixArrayWithRMQ : public SuffixArray
{
    typedef SuffixArray super;

public:
    SuffixArrayWithRMQ(const string &str)
        :super(str)
    {
        RMQmin_.SetData(Height_);
    }

    SuffixArrayWithRMQ() = default;
    virtual ~SuffixArrayWithRMQ() {}

    virtual void SetupSuffixArray(const string &str)
    {
        super::SetupSuffixArray(str);
        RMQmin_.SetData(Height_);
    }

private:
    virtual int LCP(int index1, int index2)
    {
        int first = Rank_[index1];
        int second = Rank_[index2];
        if (first > second) swap(first, second);
        return RMQmin_.MostValue(first+1, second);
    }

private:
    MyRMQst<> RMQmin_;
};

class BasicList 
{
    struct ListNode
    {
        int val;
        ListNode *next;
        ListNode(int x) : val(x), next(NULL) {}
    };
public:
    ListNode* ReverseList(ListNode* head) 
    {
        ListNode *curr = head;
        ListNode *prev = NULL;
        ListNode *next = NULL;
        while(curr)
        {
            next = curr->next;
            curr->next =  prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
};

// Greatest Common Divisor
// Euclidean algorithm 
// gcd(a, b) = gcd(b, r)  
// r = a%b, k = a/b -> a = b*k +r 
// if r==0 gcd(a,b) = b  gcd(a,b)=gcd(b,a mod b)
class GCD
{
public:
    int Euclid(int a, int b){ return b == 0 ? a : Euclid(b,a%b); }

    int EuclidIte(int a, int b)
    {
        while (b!=0){
            int temp = b;
            b = a%b;
            a = temp; 
        }
        return a; // from b
    }
};

// count how many primes in n
class CountPrimes
{
public:
    int Count(int n)
    {
        //return CountOneByOne(n);
        return SieveMethod(n);
    }

private:
    bool IsPrime(int n)
    {
        if (n<2) return false;
        if (n % 2 == 0) return n == 2;
        if (n % 3 == 0) return n == 3;
        if (n % 5 == 0) return n == 5;
        for (int i = 7; i*i <= n; i += 2)
        {
            if (n%i == 0) return false;
        }
        return true;
    }

    int CountOneByOne(int n)
    {
        int count = 0;
        for (int i = 0; i<n; i++)
        {
            if (IsPrime(i)) count++;
        }
        return count;
    }

    int SieveMethod(int n)
    {
        if (n<2) return 0;
        vector<char> filter(n, 1);
        filter[0] = 0;
        filter[1] = 0;
        for (int i = 2; i*i <= n; i++)
        {
            if (filter[i] == 1)
            {
                // from i*i
                for (int j = i*i; j <= n; j += i) filter[j] = 0;
            }
        }
        int count = 0;
        for (auto val : filter)
        {
            if (val == 1) count++;
        }
        return count;
    }
};

// methods of sqr/power
class PowSqrSolution 
{
public:
    double Pow(double x, int n)
    {
        double result = 1.0;
        if (n == 0) return result;
        // x^n = x^(n/2)*x^(n/2)*x^(n%2)
        for (int i = n; i != 0; i /= 2)
        {
            if (i % 2 != 0)
                result *= x;
            x *= x;
        }

        return n>0 ? result : (1.0 / result);
    }

    int SqrtNewtonIteration(int x)
    {
        //x(n+1)=x(n)-f(x(n))/f'(x(n))
        //x(n+1)=x(n)-(x^2-n)/2x
        // xn1 = (x+n/x)/2
        if (x == 0) return 0;
        const double EPSIL = 0.001;
        double xn = 0.0, xn1 = 1.0;
        while (abs(xn1 - xn)>EPSIL)
        {
            xn = xn1;
            xn1 = (xn + x / xn) / 2.0;
        }
        return xn1;
    }

    int SqrtBinarySearch(int x)
    {
        if (x<2) return x;

        int begin = 1, end = x / 2;
        int middle = 0, last_middle = 0;
        while (begin <= end)
        {
            middle = begin + (end - begin) / 2;
            int guess = x / middle;
            if (middle == guess) return middle;

            if (middle < guess)
            {
                begin = middle + 1;
                last_middle = middle;
            }
            else
            {
                end = middle - 1;
            }
        }
        return last_middle;
    }
};

/*
  Count sort,big O is n,  bitmap sort is to save memory 
*/
class BitmapSort
{
    #define ONE_BYTE 8
public:
    void Sort(int *data,int len)
    {
        char bit_array[4*1024]={0};
        for(int i=0;i<len;i++)
        {
            int num=data[i];
            int index= num/ONE_BYTE;
            int offset= num%ONE_BYTE;
            bit_array[index] |= 1<<offset;
        }
        int array_idx=0;
        for(int i=0;i<sizeof(bit_array);i++)
        {
            for(int j=0;j<ONE_BYTE;j++)
            {
                if((bit_array[i] & 1<<j) > 0 )
                    data[array_idx++]=i*ONE_BYTE+j;
            }
        }
    }
   
};

  /*
  BFS(G, s)
    for each vertex u from G.V - {s}
        u.color = WHITE
        u.dist = infinit
        u.node = NIL
    s.color = GRAY
    s.dist = 0
    s.prev = NIL
    Q = 0
    ENQUEUE(Q,s)
    while Q != 0
        u = DEQUEUE(Q)
        for each v from u.adj[]
            if v.color == WHITE
                v.color = GRAY
                v.dist = u.dist + 1
                v.prev = u
                ENQUEUE(Q,v) 
        u.color = BLACK

  DFS(G)
    for each vertex u from G.v - {s}
        u.color = WHITE
        u.prev = NIL
    time_ = 0
    for each vertex u from G.v
        if u.color == WHITE
            DFS-VISIT(G,u)

  DFS-VISIT(G,u,time)
    time_ ++
    u.discove_time = time
    u.color = GRAY
    for each v from u.adj[]
        if v.color == WHITE
            v.prev = u
            DFS-VISIT(G,v,time)
    u.color = BLACK
    time_++
    u.finish_time = time_
    //for topological sort
    Add u onto front of topology list 

  // print path s to v
  PRINT-PATH(G,s,v) 
    if v == s
        print s
    else if v.prev == NIL
        print ¡°no path from¡± s ¡°to¡± v ¡°exists¡±
    else PRINT-PATH(G,s,v.prev)
        print v
    */
class MyGraph
{
    struct Vertex
    {
        string Id;
        int Data;
        
        Vertex(string id,int data):
            Id(id),
            Data(data){}
        ~Vertex(){}
    };

    typedef shared_ptr<Vertex> VertexPtr;

public:
    void AddVertex(const string &id,int data)
    {
        Vertexes_.insert(pair<string, VertexPtr>(id,make_shared<Vertex>(id,data)));
    }
    
    bool AddEdge(const string &from_id,const string &to_id)
    {
        auto from_v = Vertexes_.find(from_id);
        auto to_v = Vertexes_.find(to_id);
        if(from_v == Vertexes_.end() || to_v == Vertexes_.end()) return false;

        auto from_v_adj = VertexAdjacents_.find(from_id);
        if(from_v_adj == VertexAdjacents_.end())
        {
            list<VertexPtr> temp_list;
            temp_list.push_back(to_v->second);
            VertexAdjacents_[from_id]=temp_list;
        }
        else
        {
            from_v_adj->second.push_back(to_v->second);
        }

        auto to_v_adj = VertexAdjacents_.find(to_id);
        if(to_v_adj == VertexAdjacents_.end())
        {
            list<VertexPtr> temp_list;
            temp_list.push_back(from_v->second);
            VertexAdjacents_[to_id]=temp_list;
        }
        else
        {
            to_v_adj->second.push_back(from_v->second);
        }
        return true;
    }

    void SearchPath(const string &from_id, const string &to_id)
    {
        vector<string> path;
        set<string> visit_log;
        //DFS(from_id,to_id, visit_log, path); recursion version
        //DFS(from_id, to_id,path); // stack version
        BFS(from_id, to_id, path);
        if (path.size() > 0)
        {
            for (auto &val : path) 
            { 
                cout << val;
                if (to_id != val)
                {
                    cout << "->";
                }
            }
            cout << endl;
        }
        else
        {
            cout << "No path exist!" << endl;
        }
    }

    void SearchPathAll(const string &from_id, const string &to_id)
    {
        vector<string> path;
        vector<vector<string>> paths;
        set<string> visit_log;
        DFSAll(from_id, to_id, visit_log, path, paths); //recursion version
        //DFSAll(from_id, to_id, paths); // stack version
        if (paths.size() > 0)
        {
            for (auto &val : paths)
            {
                for (auto &val2 : val)
                {
                    cout << val2;
                    if (val2 != to_id) 
                        cout << "->";
                }
                cout << endl;
            }
        }
        else
        {
            cout << "No path exist!" << endl;
        }
    }

private:
    // recursion version 
    bool DFS(const string &cur_id, const string &des_id, set<string> &visit_log, vector<string> &path)
    {
        auto cur_adj = VertexAdjacents_.find(cur_id);
        if (cur_adj == VertexAdjacents_.end()) return false;

        visit_log.insert(cur_id);
        path.push_back(cur_id);
        
        if (cur_id == des_id) return true;

        for(auto val : cur_adj->second)
        {
            string vex_id = val->Id;
            if (visit_log.find(vex_id) == visit_log.end())
            {
                if (DFS(vex_id, des_id, visit_log, path)) 
                    return true;
            }
        }
        if (path.size() > 0)
        {
            path.erase(--path.end());
        }
        return false;
    }

    void DFSAll(const string &cur_id, const string &des_id, set<string> &visit_log, 
        vector<string> &path, vector<vector<string>> &paths)
    {
        auto cur_adj = VertexAdjacents_.find(cur_id);
        if (cur_adj == VertexAdjacents_.end()) return;

        visit_log.insert(cur_id);
        path.push_back(cur_id);

        if (cur_id == des_id)
        {
            paths.push_back(path);
        }

        for (auto val : cur_adj->second)
        {
            string vex_id = val->Id;
            if (visit_log.find(vex_id) == visit_log.end())
            {
                DFSAll(vex_id, des_id, visit_log, path, paths);
            }
        }
        // back-trace node
        visit_log.erase(visit_log.find(cur_id));
        if (path.size() > 0)
        {
            path.erase(--path.end());
        }
    }

    // non-recursion stack version
    bool DFS(const string &start_id, const string &des_id, vector<string> &path)
    {
        stack<string> search_path;
        set<string> visit_log;
        search_path.push(start_id);
        visit_log.insert(start_id);
        path.push_back(start_id);
        if (start_id == des_id) return true;
        bool if_found = false;

        while (search_path.size()>0)
        {
            const string &curr_id = search_path.top();
            
            auto curr_adj = VertexAdjacents_.find(curr_id);
            bool any_further = false;
            if (curr_adj != VertexAdjacents_.end())
            {
                for (auto val : curr_adj->second)
                {
                    string vex_id = val->Id;
                    if (visit_log.find(vex_id) == visit_log.end())
                    {
                        search_path.push(vex_id);
                        path.push_back(vex_id);
                        visit_log.insert(vex_id);
                        if (vex_id == des_id)
                        {
                            if_found = true;
                        }
                        // no more loog go to deeper search
                        any_further = true;
                        break;
                    }
                }
                if (if_found)
                    break;
            }
            if (!any_further)
            {
                search_path.pop();
                path.erase(--path.end());
            }
        }
        return if_found;
    }

    // !!! not right
    void DFSAll(const string &start_id, const string &des_id, vector<vector<string>> &paths)
    {
        stack<string> search_path;
        set<string> visit_log;
        vector<string> path;
        search_path.push(start_id);
        visit_log.insert(start_id);
        path.push_back(start_id);
        if (start_id == des_id) return;
        while (search_path.size()>0)
        {
            const string &curr_id = search_path.top();
            
            auto curr_adj = VertexAdjacents_.find(curr_id);
            bool any_further = false;
            
            if (curr_adj != VertexAdjacents_.end())
            {
                for (auto val : curr_adj->second)
                {
                    
                    string vex_id = val->Id;
                    if (visit_log.find(vex_id) == visit_log.end())
                    {
                        search_path.push(vex_id);
                        visit_log.insert(vex_id);
                        path.push_back(vex_id);
                        if (vex_id == des_id)
                            paths.push_back(path);

                        any_further = true;
                        break;
                    }
                }  
            }
            if (!any_further)
            {
                search_path.pop();
                path.erase(--path.end());
            }
        }
    }

   
    // !!! not right BFS based on a queue, find the shortest path
    bool BFS(const string &from_id, const string &des_id, vector<string> &path)
    {
        typedef struct search_info
        {
            string id;
            int depth;

            search_info(const string &str,int n)
                :id(str),
                depth(n){}
        }search_info_type;

        int curr_level = 0;
        set<string> visit_log;
        queue<search_info_type> search_path;
        search_path.push(search_info_type(from_id, curr_level));
        visit_log.insert(from_id);
        
        bool if_found = false;
        if (from_id == des_id) return true;

        while (!search_path.empty())
        {
            const search_info_type &search_info = search_path.front();
            const string &curr_id = search_info.id;
            int prev_level = curr_level;
            curr_level = search_info.depth;
            if (path.size() > 0 && curr_level == prev_level)
            {
                path.erase(--path.end());
            }
            path.push_back(curr_id);
            visit_log.insert(curr_id);
            
            auto adj = VertexAdjacents_.find(curr_id);
            if (adj != VertexAdjacents_.end())
            {
                for (auto &val : adj->second)
                {
                    // check adjacent vertex
                    const string &vex_id = val->Id;
                    if (visit_log.find(vex_id) == visit_log.end())
                    {
                        search_path.push(search_info_type(vex_id, curr_level+1));
                        if (vex_id == des_id)
                        {
                            if_found = true;
                            path.push_back(vex_id);
                            break;
                        }
                    }
                }
                if (if_found)
                    break;
            }
            search_path.pop();
        }
        return if_found;
    }

public:
    void ShowEdges()
    {
        for (auto val : VertexAdjacents_)
        {
            cout << val.first << "->";
            for (auto val2 : val.second)
            {
                cout << val2->Id;
            }
            cout << endl;
        }
    }

private:
    map<string, VertexPtr> Vertexes_;
    map<string, list<VertexPtr>> VertexAdjacents_;
};

/*
Basic Dynamic Programming 
*/
class BasicDP
{
public:
    // longest increasing subsequence O(N^2)
    int LIS(const vector<int> &input)
    {
        if (input.size() == 0) return 0;

        vector<int> lis_val;        
        int longest_val = 1;
        for_each(input.begin(), input.end(), [&](auto val) 
        { 
            lis_val.push_back(1);  
        });

        for (int i = 0; i < input.size(); i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (input[i] >= input[j] && lis_val[j] + 1 > lis_val[i])
                    lis_val[i] = lis_val[j] + 1;
            }
            if (lis_val[i] > longest_val) longest_val = lis_val[i];
        }
        return longest_val;
    }

    vector<int> LISVector(const vector<int> &input)
    {
        if (input.size() == 0) return vector<int>();

        vector<vector<int>> lis_val;
        int longest_val = 1;
        int longest_index = 0;
        int selected_index = 0;
        for (int i = 0; i < input.size(); i++)
        {
            vector<int> temp;
            temp.push_back(input[0]);
            lis_val.push_back(temp);
        }

        for (int i = 0; i < input.size(); i++)
        {
            bool selected_updated = false;
            for (int j = 0; j < i; j++)
            {
                if (input[i] >= input[j] && lis_val[j].size() + 1 > lis_val[i].size())
                {
                    selected_index = j;
                    selected_updated = true;
                }
            }
            if (selected_updated)
            {
                vector<int> temp_vec(lis_val[selected_index]);
                temp_vec.push_back(input[i]);
                lis_val[i].swap(temp_vec);
            }
            if (lis_val[i].size() > longest_val)
            {
                longest_val = lis_val[i].size();
                longest_index = i;
            }
        }
        return lis_val[longest_index];
    }

    // 0-1 KnapSack 
    int Knapsack0_1(const vector<int> &value_tab, const vector<int> &weight_tab, int weight_limit)
    {
        vector<vector<int>> max_value_tab;
        for (int i = 0; i < value_tab.size(); i++)
        {
            vector<int> temp;
            for (int j = 0; j <= weight_limit; j++)
            {
                temp.push_back(0);
            }
            max_value_tab.push_back(temp);
        }

        for (int index = 0; index < value_tab.size(); index++)
        {
            int weight_item = weight_tab[index];
            int value_item = value_tab[index];
            int pre_index = 0;
            if (index > 0)
                pre_index = index - 1;

            for (int j = 0; j <= weight_limit; j++)
            {    
                int  without_item_value = max_value_tab[pre_index][j];
                int max_value = without_item_value;
                if (weight_item <= j)
                {
                    int with_item_value =  0;
                    if (index>0)
                    {
                        int wighout_item_weight = j - weight_item;
                        with_item_value = max_value_tab[pre_index][wighout_item_weight] + value_item;
                    }
                    else
                    { 
                        with_item_value = value_item;
                    }
                    max_value = without_item_value>with_item_value ? without_item_value : with_item_value;
                }
                max_value_tab[index][j] = max_value;
            }
        }

        return max_value_tab[value_tab.size() - 1][weight_limit - 1];
    }
};

template<typename id_type,typename data_type>
struct MyTreeNode
{
public:
    id_type id;
    data_type data;
    MyTreeNode *left;
    MyTreeNode *right;
    MyTreeNode(id_type in_id, data_type in_data) :
        id(in_id),
        data(in_data),
        left(nullptr),
        right(nullptr) {}
    ~MyTreeNode() = default;
};

template<typename id_type, typename data_type>
class MyBinaryTree
{
    typedef MyTreeNode<id_type, data_type> BinaryTreeNode;
    typedef shared_ptr<BinaryTreeNode> SharedPtrTreeNode;
    typedef function< void(BinaryTreeNode*)> VisitFunc;
public:

    typedef enum
    {
        ChildLeft,
        ChildRight
    }ChildType;

public:
    
    BinaryTreeNode* AddTreeNode(id_type in_id, data_type in_data, BinaryTreeNode *parent=nullptr, ChildType child_type = ChildLeft)
    {
        auto ptn = make_shared<BinaryTreeNode>(in_id, in_data);
        _tree.insert(pair<id_type, SharedPtrTreeNode>(in_id, ptn));
        if (parent)
        {
            if (child_type == ChildLeft) parent->left = ptn.get();
            else parent->right = ptn.get();
        }
        return ptn.get();
    }
    
    bool AddChild(id_type parent_id, id_type child_id, ChildType child_type = ChildLeft)
    {
        auto p_parent = _tree.find(parent_id);
        auto p_child = _tree.find(child_id);
        if (p_parent == _tree.end() || p_child == _tree.end()) return false;

        AddChild(p_parent->second.get(),p_child->second.get(),child_type);
        return true;
    }

    void AddChild(BinaryTreeNode *parent, BinaryTreeNode *child, ChildType child_type = ChildLeft)
    {
        if (parent)
        {
            if (child_type == ChildLeft) parent->left = child;
            else parent->right = child;
        }
    }

    BinaryTreeNode* FindNode(id_type id)
    {
        auto ptn = _tree.find(id);
        if (ptn != _tree.end()) return ptn->second.get();
        else return nullptr;
    }

    void TraversePreOrder(BinaryTreeNode* root, VisitFunc visit_func)
    {
        if (root == nullptr) return;
        visit_func(root);
        TraversePreOrder(root->left,visit_func);
        TraversePreOrder(root->right, visit_func);
    }

private:
    map<id_type, SharedPtrTreeNode> _tree;
};

template<typename id_type, typename data_type>
class LCA
{
    typedef MyTreeNode<id_type, data_type> BinaryTreeNode;

public:
    // return lowest common ancestor
    BinaryTreeNode *BruteRecursive(BinaryTreeNode *current, BinaryTreeNode *node1, BinaryTreeNode *node2)
    {
        if (current == nullptr || current == node1 || current == node2) return current;

        BinaryTreeNode *left = BruteRecursive(current->left, node1, node2);
        BinaryTreeNode *right = BruteRecursive(current->right, node1, node2);

        if (left != nullptr && right != nullptr) 
            return current;
        else if (left != nullptr) 
            return left;
        else if (right != nullptr) 
            return right;
        return 
            nullptr;
    }

    BinaryTreeNode* BinarySearchTree(BinaryTreeNode *root, BinaryTreeNode *node1, BinaryTreeNode *node2) 
    {
        if(root == nullptr || node1==nullptr || node2==nullptr) 
            return nullptr;
        
        if(root->data > node1->data && root->data > node2->data) 
            return BinarySearchTree(root->left,node1,node2);
        if(root->data < node1->data && root->data < node2->data) 
            return BinarySearchTree(root->right,node1,node2);
        else 
            return root;
    }
};

#endif
