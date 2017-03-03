#include "generic.hpp"
#include "leetcode.hpp"

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

int main(int argc,char *argv[])
{
    TestCombineKoutofN();
    //TestWordLadder();
    //MyGraph::Test();
    //BasicDP::TestKnapsack01();
    ::getchar();
    return 0;
}
