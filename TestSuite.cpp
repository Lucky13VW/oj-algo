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

static void TestGrayCode()
{
    int n = 3;
    GrayCode gray_code;
    vector<int> &&result = gray_code.Solution(n);
    
    for_each(result.begin(), result.end(), [&](auto val) 
    { 
        int index = 1 << (n-1);
        for (int i = 0; i < n; i++)
        {
            int out_put = 0;
            if ((index & val) > 0) out_put = 1;
            cout << out_put;
            index = index >> 1;
        } 
        cout << endl;
    });
}

static void TestEditDistance()
{
    EditDistance sln;
    string str1 = "abc";
    string str2 = "ebcf";
    int dist = sln.minDistance(str1,str2);
    cout << "Str1: " << str1 << endl;
    cout << "Str2: " << str2 << endl;
    cout <<"MinDist:" << dist << endl;
}

static void TestRandomList()
{
    RandomListNode *head = NULL,*curr=NULL;
    RandomListNode *node1 = new RandomListNode(1);
    RandomListNode *node2 = new RandomListNode(2);
    node1->next = node2;
    RandomListNode *node3 = new RandomListNode(3);
    node2->next = node3;
    RandomListNode *node4 = new RandomListNode(4);
    node3->next = node4;
    node3->random = node3;
    node2->random = node4;
    function<void()> func_print = [&]() 
    {
        while (curr)
        {
            cout << curr->label << ":" << (curr->random == NULL ? -1:curr->random->label) << " ";
            curr = curr->next;
        }
    };
    head = node1;
    curr = head;
    func_print();
    cout << endl;
    CopyRandomList sln;
    RandomListNode *new_head = sln.CopyList(head);
    curr = new_head;
    func_print();
    cout << endl;
}

void TestNumberOfIslands()
{
    vector<string> input
    { "11111011111111101011","01111111111110111110","10111001101111111111","11110111111111111111",
    "10011111111111111111","10111111011101110111","01111111111101101111","11111111111101111011",
    "11111111110111111111","11111111111111111111","01111111011111111111","11111111111111111111",
    "11111111111111111111","11111011111110111111","10111110111011110111","11111111111101111110",
    "11111111111110111100","11111111111111111111","11111111111111111111","11111111111111111111" };

    NumberOfIslands sln;
    vector<vector<char>> grid;
    function<void(const string &)> proc = [&](auto str) 
    { 
        vector<char> line;
        for (auto c : str)
        {
            line.push_back(c);
            cout << c;
        }
        grid.push_back(line); 
        cout << endl;
    };
    
    for(auto &val: input)
        proc(val);
    
    cout << "Islands:" <<sln.CountIslands(grid) << endl;
    for (int i = 0; i < grid.size(); i++)
    {
        for (int j = 0; j < grid[0].size(); j++) cout << grid[i][j];
        cout << endl;
    }
}

int main(int argc,char *argv[])
{
    TestNumberOfIslands();
    //TestRandomList();
    //TestEditDistance();
    //TestGrayCode();
    //TestBinaryTreeLCA();
    //TestCombineKoutofN();
    //TestWordLadder();
    //TestMyGraph();
    //TestKnapsack01();
    ::getchar();
    
    return 0;
}