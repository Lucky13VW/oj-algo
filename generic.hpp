#ifndef GENERIC_HPP
#define GENERIC_HPP

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <list>
#include <set>
#include <memory>
#include <algorithm>

using namespace std;

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
    
    static void Test()
    {
        int data[]={1,17,25,30,28,14,3,12,9,7,6,10,5,4,2,8};
        int len = sizeof(data)/sizeof(int);
        for(int i=0;i<len;i++) cout<<data[i]<<" ";
        cout<<endl<<"Sorted:"<<endl;
        BitmapSort bm;
        bm.Sort(data,len);
        for(int i=0;i<len;i++) cout<<data[i]<<" ";
        cout<<endl;
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
        DFS(from_id,to_id, visit_log, path);
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

private:
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
                if (DFS(vex_id, des_id, visit_log, path)) return true;
            }
        }
        if (path.size() > 0)
        {
            path.erase(--path.end());
        }
        return false;
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

    static void Test()
    {
        MyGraph graph;
        graph.AddVertex("A",1);
        graph.AddVertex("B",2);
        graph.AddVertex("C",3);
        graph.AddVertex("D",4);
        graph.AddVertex("E",5);
        graph.AddVertex("F",6);
        graph.AddEdge("A","B");
        graph.AddEdge("A","C");
        graph.AddEdge("B","D");
        graph.AddEdge("C","D");
        graph.AddEdge("C","E");
        graph.AddEdge("B","F");
        graph.ShowEdges();
        cout << "From A to E" << endl;
        graph.SearchPath("A","E");
        cout << "From B to E " << endl;
        graph.SearchPath("B","E");
        cout << "From E to F" << endl;
        graph.SearchPath("E","F");
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

#endif