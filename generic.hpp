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

using namespace std;

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x)
        : val(x), next(NULL) {}
};

class BasicList 
{
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
        print ��no path from�� s ��to�� v ��exists��
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

class Combinate
{
public:
  
    void TotalC(int arr[], int count)
    {
        int total_num = (1<<count)-1;
        for(int i=1; i< total_num+1;i++)
        {
            int j = 1,idx=0;
            vector<int> disp;
            while(j<=i)
            {
                if(j&i)
                {
                    disp.push_back(arr[idx]);
                }
                j<<=1;
                idx++;
            }
            PrintList(disp);
        }
    }
    
private:
    void PrintList(const vector<int>&vec)
    {
        vector<int>::const_iterator iter=vec.begin();
        for(;iter!=vec.end();iter++)
        {
            cout<<*iter<<" ";
        }
        cout<<endl;
    }

public:
    static void Test()
    {
        const int count=3;
        int arr[count]={1,2,3};
        Combinate comb;
        comb.TotalC(arr,count);
    }
};

class Permutate
{
public:
    
    void TotalP(int arr[],int count,int index,int need=0)
    {
        if (need == 0)
            need = count;
                               
        if( index == need)
        {
            PrintArr(arr,need);
            return;
        }
        
        for(int i=index;i<count;i++)
        {
            Swap(arr[i],arr[index]);
            TotalP(arr,count,index+1,need);
            Swap(arr[i],arr[index]);
        }
    }

    void TotalPStl(int arr[], int count)
    {
        int total = 0;
        bool is_ok = false;
        do
        {
            is_ok = next_permutation(arr,arr+count);
            PrintArr(arr,count);
            total++;
        }while(is_ok);

        while(prev_permutation(arr,arr+count))
        {
            PrintArr(arr,count);
            total++;
        }
        cout<<"total:"<<total<<endl;
    }
    
private:
    void PrintArr(int arr[],int count)
    {
         for(int j=0;j<count;j++)
             cout<<arr[j]<<" ";
         cout<<endl;
    }

    void Swap(int &arr1, int &arr2)
    {
        if (arr1 != arr2)
            swap(arr1,arr2);
    }
    
public:
    static void Test()
    {
        const int count = 3;
        int data[count]={1,2,3};
        Permutate per;
        per.TotalP(data,count,0,3);
        //per.TotalPStl(data,count);
    }  
};

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

class ComparerMin
{
public:
    static int Value(int a, int b) { return min(a, b); }
};
template<class Comparer= ComparerMin>
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
        int range = int(log(nums.size())/log(2))+1;
        int num_size = nums.size()+1;
        vector<int> temp(range, 0);
        Data_.clear();
        Data_.resize(num_size,temp);

        for (int i = 1; i < num_size; i++) Data_[i][0] = nums[i-1];

        for (int j = 1; j < range; j++)
        {
            for (int i = 1; i < num_size; i++)
            {
                int k = i + (1 << j) - 1;
                if (k < num_size)
                    Data_[i][j]= Comparer::Value(Data_[i][j - 1], Data_[i + (1 << (j - 1))][j - 1]);
            }
        }
    }

    int MostValue(int i, int j)
    {
        i++; j++;
        if (i > j) swap(i, j);

        // k=log2(j-i+1) 
        int k = int(log(j - i + 1) / log(2));
        // RMQ(i,j) = min(F[i,k], F[j-2^k+1,k])
        return Comparer::Value(Data_[i][k], Data_[j + 1 - (1 << k)][k]);
    }

private:
    vector<vector<int>> Data_;
};

class MySuffixArray
{

public:
    MySuffixArray(const string &str)
    {
        SetString(str);
    }
    MySuffixArray() = default;
    ~MySuffixArray() = default;
    
    void SetString(const string &str)
    {
        RawString_ = str;
        SA_.clear();
        for (int i = 0; i < str.size(); i++)
        {
            SA_.push_back(i);
        }
        sort(SA_.begin(), SA_.end(), [&](int offset1, int offset2) 
        { return RawString_.substr(offset1) < RawString_.substr(offset2); });

        // calculate rank, the raw index in SA
        Rank_.resize(SA_.size(),0);
        for (int i = 0; i < SA_.size(); i++)
        {
            Rank_[SA_[i]] = i;
        }
        // calculate height array height[i] = LCP(i,i-1)
        Height_.resize(SA_.size(), 0);
        for (int i = 1; i < SA_.size(); i++)
        {
            Height_[i] = TwoStringLCP(SuffixStr(i) , SuffixStr(i-1));
        }
        RMQmin_.SetData(Height_);
    }

    int Rank(const string &pat)
    {
        int lo = 0;
        int hi = SA_.size() - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            const string &suffix = SuffixStr(mid);
            if (pat < suffix)  hi = mid - 1;
            else if (pat > suffix) lo = mid + 1;
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
            const string &suffix = SuffixStr(mid);
            // search for prefix matched string in suffix
            int res = suffix.compare(0,pat.size(),pat);
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
        return SuffixStr(index).substr(0, Height_[index]);
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
        for (int i = 1; i < RawString_.size()/2; i++)
        {
            // odd i is the center of palindromic string LCP(i,len-i-1) 
            // xx|i.......|(len-i-1)xxx
            int lcp = LCP(i, RawString_.size() - i -1);
            if(lcp*2-1 > max)
            {
                max = lcp*2-1;
                index = i-lcp+1;
            }
            // even xx|i.......|(len-i)xx
            lcp = LCP(i, RawString_.size() - i );
            if (lcp*2 > max)
            {
                max = lcp*2;
                index = i-lcp;
             }
        }
        return RawString_.substr(index, max);
    }

private:
    string SuffixStr(int sa_index)
    {
        return RawString_.substr(SA_[sa_index]);
    }

    // lcp(RawString[index1],RawString[index2])
    int LCP(int index1, int index2)
    {
        int first = Rank_[index1];
        int second = Rank_[index2];
        if (first > second) swap(first,second);

        int min = INT_MAX;
        for (int i = first+1; i <= second; i++) 
        {
            if (min > Height_[i]) min = Height_[i];
        }
        return min;
    }

    int LCPRMQ(int index1, int index2)
    {
        return RMQmin_.MostValue(Rank_[index1], Rank_[index2]);
    }

    int TwoStringLCP(const string &str1,const string &str2)
    {
        int len = min(str1.size(),str2.size());
        for (int i = 0; i < len; i++)
        {
            if (str1[i] != str2[i]) return i;
        }
        return  len;
    }

   
private:
    vector<int> SA_;
    vector<int> Rank_;
    vector<int> Height_;
    string RawString_;
    MyRMQst<> RMQmin_;
};

// string LSD(least significant digit) sort
class LSDSort
{
public:
    void Sort(vector<string> &StrArr)
    {
        if (StrArr.size() == 0) return;

        int Radix = 256;
        
        int str_len = StrArr[0].size();
        int arr_len = StrArr.size();
        // from right to left LSD
        for (int d = str_len - 1; d >= 0; d--)
        {
            vector<int> count(Radix+1, 0);
            // compute frequency of each character, count[r+1!!!]  
            for (int i = 0; i < arr_len; i++) 
                count[StrArr[i].at(d) + 1]++;

            // change counts to indices
            for (int i = 0; i < Radix; i++) 
                count[i + 1] += count[i];

            // distribute it to temp array
            // r = count[StrArr[i].at(d)]++ 
            // '++' here is to handle same character, move r to next position in aux
            vector<string> aux(arr_len,"");
            for (int i = 0; i < arr_len; i++) 
                aux[count[StrArr[i].at(d)]++] = StrArr[i]; 

            // save back to string array
            for (int i = 0; i < arr_len; i++)
                StrArr[i] = aux[i];
        }
    }
};

#endif
