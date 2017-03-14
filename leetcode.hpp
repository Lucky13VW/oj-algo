#ifndef LEETCODE_HPP
#define LEETCODE_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <stack>
#include <algorithm>
#include <string>

using namespace std;

/*
1. Two Sum
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
*/
class TwoSumSolution {
public:
    vector<int> twoSum(vector<int>& nums, int target) 
    {
        unordered_map<int,int> value_map;
        
        for(int i=1;i<nums.size();i++)
        {
            value_map[nums[i]]=i;
        }
        vector<int> result;
        for(int i=0;i<nums.size();i++)
        {
            auto iter = value_map.find(target-nums[i]);
            if(iter != value_map.end())
            {
                result.push_back(i+1);
                result.push_back(iter->second+1);
                break;
            }
        }
        return result;
    }
};


/*
2. Add Two Numbers
You are given two linked lists representing two non-negative numbers. 
The digits are stored in reverse order and each of their nodes contain a single digit. 
Add the two numbers and return it as a linked list.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
*/
class AddTwoNumbersSolution {

   struct ListNode {
      int val;
      ListNode *next;
      ListNode(int x) : val(x), next(NULL) {}
  }; 

public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) 
    {
        ListNode*iter1=l1,*iter2=l2;
        stack<int> my_stack;
        while(iter1!=NULL || iter2!=NULL)
        {
            int input1= 0, input2 = 0;
            if(iter1!=NULL)
            {
                input1=iter1->val;
                iter1=iter1->next; 
            }
            if(iter2!=NULL)
            {
                input2=iter2->val;
                iter2=iter2->next;
            }
            int value= input1 + input2;
            my_stack.push(value);
        }
        ListNode *result=NULL,*prev=NULL;
        int add=0;
        while(!my_stack.empty())
        {
            int num = my_stack.top()+add;
            add = 0;
            my_stack.pop();
            if(num>9) 
            {
                num-=10; 
                add=1;
            }
            ListNode *cur = new ListNode(num);
            
            if(result==NULL) result=cur;
            if(prev!=NULL) 
                prev->next = cur;
            prev=cur;
        }
        if(add == 1)
        {
           ListNode *add_one = new ListNode(1);
           prev->next = add_one;
        }
        return result;
    }
};

/*
8. String to Integer (atoi)
*/
class MyAtoiSolution {
public:
    int myAtoi(string str) 
    {
        int total=0;
        int sign=1;
        size_t start = str.find_first_not_of(" ");
        for(int i=start;i<str.size();i++)
        {
            char val =str[i];
            if(val >='0' && val<='9')
            {
                if(total> INT_MAX/10 || (total==INT_MAX/10 && val-'0'>INT_MAX%10))
                     return sign==1?INT_MAX:INT_MIN;
                total=total*10+(val-'0');
            }
            else if('-' == val && i==start) sign=-1;
            else if('+'==val && i==start) continue;
            else break;
        }
        
        return total*sign;
    }
};

/*
44. Wildcard Matching
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") ¡ú false
isMatch("aa","aa") ¡ú true
isMatch("aaa","aa") ¡ú false
isMatch("aa", "*") ¡ú true
isMatch("aa", "a*") ¡ú true
isMatch("ab", "?*") ¡ú true
isMatch("aab", "c*a*b") ¡ú false
*/

class WildCardMatch
{
public:
    bool IsMatch(string s, string p)
    {
        
    }
};

/*
77. Combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
*/
class CombinationSolution 
{
public:
    vector<vector<int>> combine(int n, int k) 
    {
        vector<vector<int>> result;
        vector<int> group;
        int start = 0;
        dfs(n,k,1,0,group,result);
        return result;
    }

    vector<vector<int>> combine2(int n, int k) 
    {
        vector<vector<int>> result;
        int i = 0;
        vector<int> group(k, 0);
        while (i >= 0) 
        {
            group[i]++; // Increment element at index i

            if (group[i] > n) // Move index to the left if the element exceeded n.
            {
                --i;
            }
            /* If the index is at the end of the vector
            * c, then (because the other conditions are
            * obeyed), we know we have a valid combination,
            * so push it to our ans vector<vector<>>
            */
            else if (i == k - 1)
            {
                result.push_back(group);
            }
            /* Move index to the right and set the
            * element at that index equal to the
            * element at the previous index.
            *
            * Because of the increment at the beginning
            * of this while loop, we ensure that the
            * element at this index will be at least
            * one more than its neighbor to the left.
            */
            else 
            {
                ++i;
                group[i] = group[i - 1];
            }
        }
        return result;
    }

private:
    
    void dfs(int n, int k,int start,int count,vector<int> &group,vector<vector<int>> &result)
    {
        if(count ==k)
        {
            result.push_back(group);
        }
        else
        {
            for(int i=start;i<=n;i++)
            {
                group.push_back(i);
                dfs(n,k,i+1,count+1,group,result);
                group.pop_back();
            }
        }
    }
};

/*
89 Gray Code
The gray code is a binary numeral system where two successive values differ in only one bit.
Given a non-negative integer n representing the total number of bits in the code, 
print the sequence of gray code. A gray code sequence must begin with 0.
For example, given n = 2, return [0,1,3,2].
*/
class GrayCode
{
public:
    /*
    G(i) = i^ (i>>1).
    */
    vector<int> Solution(int n)
    {
        vector<int> result;
        for (int i = 0; i < (1 << n); i++)
        {
            result.push_back(i ^  (i >> 1));
        }
        return result;
    }
};

/*
121. Best Time to Buy and Sell Stock  QuestionEditorial Solution  My Submissions
Total Accepted: 112522
Total Submissions: 307050
Difficulty: Easy
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5
max. difference = 6-1 = 5 (not 7-1 = 6£©

Example 2:
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.
*/

class BestTimeStock1Solution
{
public:
    vector<int> MaxProfit(vector<int> &prices)
    {
        vector<int> best;
        if (prices.size() == 0)
            return best;  
          
        int try_buy_price = prices[0]; 
        int best_buy_time=0,try_buy_time=0,best_sell_time=0,max_profit = 0;
        for (int i = 1; i < prices.size(); i++)
        {  
            //prices[i] - min > profit ? prices[i] - min : profit;
            int profit = prices[i] - try_buy_price;
            if(profit > max_profit)
            {
                max_profit = profit;
                best_sell_time = i;
                best_buy_time = try_buy_time;
            }
            if( prices[i] < try_buy_price)
            {
                try_buy_price = prices[i];
                try_buy_time = i;
            }
            //prices[i] < min ? prices[i] : min;  
        }
        best.push_back(best_buy_time);best.push_back(best_sell_time);best.push_back(max_profit);
        return best;  
    }
    
    static void Test()
    {
        int data[15] = {9989,9992,9998,9997,9991,9925,9994,9993,9992,9999,9990,9989,9988,9987,9986};
        vector<int> prices;
        for (int i=0;i<15;i++)
            prices.push_back(data[i]);

        BestTimeStock1Solution best1;
        vector<int> besttime=best1.MaxProfit(prices);
        cout<<besttime[0]<<","<<besttime[1]<<","<<besttime[2]<<endl;
    }
};

class BestTimeSellStockIII
{
public:    
    int maxProfit(vector<int> &prices)
    {
        if(prices.size() ==  0) return 0;

        int max_profit=0;

        for(int i=0;i<prices.size();i++)
        {
            max_profit = max(FindMax(prices,0,i)+FindMax(prices,i,prices.size()-1),max_profit);
        }
        return max_profit;
    }
    
private:
    int FindMax(vector<int> &prices,int start, int end)
    {
        if(start == end) return 0;

        int max_profit = 0, try_buy_price = prices[start];
        for(int i=start+1;i<end+1;i++)
        {
            max_profit = max(prices[i]-try_buy_price,max_profit);
            try_buy_price = min(prices[i],try_buy_price);
        }
        return max_profit;
    }

public:
    static void Test()
    {
        int data[8]={3,5,9,4,7,6,7,8};
        vector<int> prices;
        for(int i=0;i<8;i++)
            prices.push_back(data[i]);

        BestTimeSellStockIII best;
        cout<<best.maxProfit(prices)<<endl;
    }

    
};


/*
127. Word Ladder
Given two words (beginWord and endWord), and a dictionary's word list, 
find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time
Each intermediate word must exist in the word list
For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
*/

class WordLadderSolution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>&wordList) 
    {
        typedef struct path_info
        {
            string str;
            int    depth;
        }path_info_type;

        queue<path_info_type> visit_path;
        
        int curr_depth = 1;
        bool found = false;

        path_info_type path_info;
        path_info.str = beginWord;
        path_info.depth = curr_depth;
        visit_path.push(path_info);

        while (!visit_path.empty())
        {
            const string &curr_str = visit_path.front().str;
            curr_depth = visit_path.front().depth;
            if (curr_str == endWord)
            {
                found = true;
                break;
            }
            for (auto it = wordList.begin(); it != wordList.end();)
            {
                if (IsLadderWord(curr_str, *it))
                {
                    path_info.str = *it;
                    path_info.depth = curr_depth + 1;
                    visit_path.push(path_info);
                    it = wordList.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            visit_path.pop();
        }
        return found ? curr_depth : 0;
    }
    
    bool IsLadderWord(const string &str1, const string &str2)
    {
        int diff_count = 0;
        for(int i=0;i<str1.size();i++)
        {
            if(str1.at(i)!=str2.at(i))
            {
                diff_count++;
                if(diff_count>1)
                    break;
            }    
        }
        return diff_count<2;
    }
    
};

/*
72. Edit Distance
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. 
(each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character
*/

class EditDistance{
public:
    int minDistance(string word1, string word2) 
    {
        const size_t size1 = word1.size();
        const size_t size2 = word2.size();

        vector<vector<int>> dist_matrix;
        // initialize matirx
        for (int i = 0; i <= size1; i++)
        {
            vector<int> temp;
            for (int j = 0; j <= size2; j++)
            {
                temp.push_back(0);
            }
            dist_matrix.push_back(temp);
        }
        
        for (int j = 0; j <= size2; j++)
            dist_matrix[0][j] = j;
        for (int i = 0; i <= size1; i++)
            dist_matrix[i][0] = i;

        for (int i = 1; i <= size1; i++)
        {
            for (int j = 1; j <= size2; j++)
            {
                if (word1.at(i - 1) == word2[j - 1])
                {
                    // f[i][j] = f[i-1][j-1]
                    dist_matrix[i][j] = dist_matrix[i - 1][j - 1];
                }
                else
                {
                    // min(f[i-1][j]+1,f[i][j-1]+1,f[i-1][j-1]+1)
                    int min_step = min(dist_matrix[i - 1][j], dist_matrix[i][j - 1]);
                    dist_matrix[i][j] = min(min_step, dist_matrix[i - 1][j - 1]) + 1;
                }
            }
        }

        return dist_matrix[size1][size2];
    }
};
/*
  419. Battleships in a Board
 */

class BattleshipsInBoard {
public:
    int CountBS(vector<vector<char>>& board)
    {
        int count = 0;
        for(int i=0;i<board.size();i++)
        {
            for(int j=0;j<board[i].size();j++)
            {
                if(board[i][j] != '.')
                {
                    if(i==0 || board[i-1][j] == '.')
                    {
                        if(j==0 || board[i][j-1] == '.')
                        {
                            count++;
                        }
                    }
                }
            }
        }
        return count;
    }
};

#endif
