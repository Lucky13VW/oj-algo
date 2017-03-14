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

using namespace std;

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x)
        : val(x), next(NULL) {}
};

class BasicList {
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

class PowSqrSolution {
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

    // BFS based on a queue, find the shortest path
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
        int data[count]={1,2,2};
        Permutate per;
        per.TotalP(data,count,0,3);
        //per.TotalPStl(data,count);
    }  
};

class BasicDP
{
public:
    static void TestLIS()
    {
        int arr[] = {3,8,4,8,4,6};
        vector<int>  input;
        for (int val : arr)
        {
            cout << val << " ";
            input.push_back(val);
        }
        cout << endl;
        BasicDP dp;
        cout << dp.LIS(input) << endl;
        vector<int> output = dp.LISVector(input);
        for_each(output.cbegin(), output.cend(), [](auto val) { cout << val << " ";  });
        cout << endl;
    }

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
struct TreeNode
{
public:
    id_type id;
    data_type data;
    TreeNode *left;
    TreeNode *right;
    TreeNode(id_type in_id, data_type in_data) :
        id(in_id),
        data(in_data),
        left(nullptr),
        right(nullptr) {}
    ~TreeNode() = default;
};

template<typename id_type, typename data_type>
class MyBinaryTree
{
    typedef TreeNode<id_type, data_type> BinaryTreeNode;
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
    typedef TreeNode<id_type, data_type> BinaryTreeNode;

public:
    // return lowest common ancestor
    BinaryTreeNode *BruteRecursive(BinaryTreeNode *current, BinaryTreeNode *node1, BinaryTreeNode *node2)
    {
        if (current == nullptr) return nullptr;
        if (current == node1 || current == node2) return current;

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
};

#endif
