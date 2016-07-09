#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <stack>

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
77. Combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
*/
class CombinationSolution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> result;
        vector<int> group;
        int start = 0, couont = 0;
        dfs(n,k,1,0,group,result);
        return result;
    }
    
    void dfs(int n, int k,int start,int count,vector<int> &group,vector<vector<int>> &result){
        if(count ==k){
            result.push_back(group);
        }
        else{
            for(int i=start;i<=n;i++){
                group.push_back(i);
                dfs(n,k,i+1,count+1,group,result);
                group.pop_back();
            }
        }
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
wordList = ["hot","dot","dog","lot","log"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
*/
class WordLadderSolution {
public:
    int ladderLength(string beginWord, string endWord, unordered_set<string>& wordList) 
    {
        int total=0;
        bool bingo=false;
        while(!bingo)
        {
            bool found = false;
            for(auto iter = wordList.begin();iter!=wordList.end();iter++)
            {
                int count=0;
                for(int i=0;i<beginWord.length();i++)
                {
                    if(beginWord[i]!=(*iter)[i]) 
                        count++;
                    if(count>1)
                    {
                        break;
                    }
                }
                
                if(count == 1)
                {
                    beginWord=*iter;
                    total++;
                    wordList.erase(iter);
                    found = true;
                    break;
                }
            }
            if(bingo) 
            {
                break;
            }
            if(!found)
            {
                total=0;
                break;
            }
        }
        return total;
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

int main(int argc,char *argv[])
{

    return 0;
}
