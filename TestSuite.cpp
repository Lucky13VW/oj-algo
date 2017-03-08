#include "generic.hpp"
#include "leetcode.hpp"

static void TestBitmapSort()
{
    int data[] = { 1,17,25,30,28,14,3,12,9,7,6,10,5,4,2,8 };
    int len = sizeof(data) / sizeof(int);
    for (int i = 0; i<len; i++) cout << data[i] << " ";
    cout << endl << "Sorted:" << endl;
    BitmapSort bm;
    bm.Sort(data, len);
    for (int i = 0; i<len; i++) cout << data[i] << " ";
    cout << endl;
}

static void TestCombineKoutofN()
{
    CombinationSolution combine;
    vector<vector<int>> result;
    result = combine.combine2(4, 2);
    for_each(result.begin(), result.end(), [](auto &val) 
    {
        for (int dat : val)
            cout << dat << " ";
        cout << endl;
    });
}

static void TestKnapsack01()
{
    const int array_size = 5;
    int weight_arr[array_size] = { 2,2,6,5,4 };
    int value_arr[array_size] = { 6,3,5,4,6 };

    vector<int> weight;
    vector<int> value;

    cout << "weight:" << endl;
    for (int data : weight_arr)
    {
        weight.push_back(data);
        cout << data << " ";
    }
    cout << endl << "value:" << endl;
    for (int data : value_arr)
    {
        value.push_back(data);

        cout << data << " ";
    }
    cout << endl;
    BasicDP dp;
    int max = dp.Knapsack0_1(value, weight, 10);
    cout << "total:10, max:" << max << endl;;

}

static void TestWordLadder()
{
    vector<string> dict_set;
    dict_set.push_back("hot");
    dict_set.push_back("dot");
    dict_set.push_back("dog");
    dict_set.push_back("lot");
    dict_set.push_back("log");
    dict_set.push_back("cog");
    string begin_word = "hit";
    string end_word = "cog";

    WordLadderSolution word_ladder_sln;
    cout << "Dictionary:" << endl;
    for_each(dict_set.cbegin(), dict_set.cend(), [](auto &val) { cout << val << " "; });
    cout << endl << "From:" << begin_word << endl;
    cout << "To:" << end_word << endl;
    cout << "Step:" << word_ladder_sln.ladderLength(begin_word, end_word, dict_set) << endl;
}

static void TestMyGraph()
{
    MyGraph graph;
    graph.AddVertex("A", 1);
    graph.AddVertex("B", 2);
    graph.AddVertex("C", 3);
    graph.AddVertex("D", 4);
    graph.AddVertex("E", 5);
    graph.AddVertex("F", 6);
    graph.AddVertex("G", 7);
    graph.AddEdge("A", "B");
    graph.AddEdge("A", "C");
    graph.AddEdge("B", "D");
    graph.AddEdge("C", "D");
    graph.AddEdge("C", "E");
    graph.AddEdge("B", "F");
    graph.AddEdge("G", "D");
    graph.ShowEdges();
    cout << "From A to E" << endl;
    graph.SearchPath("A", "E");
    cout << "From E to F all paths" << endl;
    graph.SearchPathAll("E", "F");
}



static void TestBinaryTreeLCA()
{
    typedef MyBinaryTree<string, int> TreeType;
    typedef LCA<string, int> LCAType;
    TreeType BinaryTree;
   
    auto root = BinaryTree.AddTreeNode("A", 0);
    BinaryTree.AddTreeNode("B", 1, root);
    BinaryTree.AddTreeNode("C", 2);
    BinaryTree.AddTreeNode("D", 3);
    BinaryTree.AddTreeNode("E", 4);
    BinaryTree.AddTreeNode("F", 5);

    BinaryTree.AddChild("A", "C", TreeType::ChildRight);
    BinaryTree.AddChild("C", "D", TreeType::ChildLeft);
    BinaryTree.AddChild("C", "E", TreeType::ChildRight);
    BinaryTree.AddChild("E", "F", TreeType::ChildLeft);
    
    BinaryTree.TraversePreOrder(root, [](auto val) { cout << val->id << val->data << " "; });
    cout << endl;
    LCAType lca_algo;

    string node1 = "C", node2 = "D";
    auto result = lca_algo.BruteRecursive(root,BinaryTree.FindNode(node1),BinaryTree.FindNode(node2));
    if (result == nullptr) cout << " failed to find LCA" << endl;
    else cout << "LCA("<< node1 << "," << node2 << "): " << result->id << " " << result->data << endl;

}

int main(int argc,char *argv[])
{
    TestBinaryTreeLCA();
    //TestCombineKoutofN();
    //TestWordLadder();
    //TestMyGraph();
    //TestKnapsack01();
    ::getchar();
    return 0;
}
