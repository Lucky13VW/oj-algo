#ifndef LEETCODE_HPP
#define LEETCODE_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <climits>
#include <stack>
#include <queue>
#include <algorithm>
#include <string>

using namespace std;

/*
1. Two Sum
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
*/
class TwoSumSolution 
{
public:
    vector<int> twoSum(vector<int>& nums, int target) 
    {
        vector<int> result;
        unordered_map<int, int> sum_map;
        for (int i = 0; i<nums.size(); i++)
        {
            sum_map[nums[i]] = i;
        }

        for (int i = 0; i<nums.size(); i++)
        {
            auto found = sum_map.find(target - nums[i]);
            if (found != sum_map.end() && found->second != i)
            {
                result.push_back(i);
                result.push_back(found->second);
                break;
            }
        }
        return result;
    }

    vector<int> twoSum2(vector<int>& nums, int target)
    {
        vector<int> result;
        for (int i = 0; i<nums.size(); i++)
        {
            int expected = target - nums[i];
            for (int j = i + 1; j<nums.size(); j++)
            {
                if (expected == nums[j])
                {
                    result.push_back(i);
                    result.push_back(j);
                    return result;
                }
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
class AddTwoNumbers 
{
public:
    ListNode* add(ListNode* l1, ListNode* l2) 
    {
        ListNode *result = NULL, *prev = NULL;

        int carry = 0;
        while (l1 != NULL || l2 != NULL)
        {
            int sum = carry;
            if (l1 == NULL)
            {
                sum += l2->val;
                l2 = l2->next;
            }
            else if (l2 == NULL)
            {
                sum += l1->val;
                l1 = l1->next;
            }
            else
            {
                sum += (l1->val + l2->val);
                l1 = l1->next;
                l2 = l2->next;
            }

            if (sum >= 10)
            {
                carry = 1;
                sum -= 10;
            }
            else carry = 0;

            ListNode *node = new ListNode(sum);
            if (prev == NULL) result = prev = node;
            else prev->next = node;
            prev = node;
        }
        if (carry == 1)
        {
            prev->next = new ListNode(carry);
        }
        return result;
    }
};

/*
445. Add Two Numbers II
You are given two non-empty linked lists representing two non-negative integers. 
The most significant digit comes first and each of their nodes contain a single digit. 
Add the two numbers and return it as a linked list.

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
*/
class AddTwoNumbersII {
public:
    ListNode* Add(ListNode* l1, ListNode* l2)
    {
        if (l1 == NULL && l2 == NULL) return NULL;

        stack<int> l1_num, l2_num;
        while (l1)
        {
            l1_num.push(l1->val);
            l1 = l1->next;
        }
        while (l2)
        {
            l2_num.push(l2->val);
            l2 = l2->next;
        }
        ListNode *node = NULL, *prev = NULL;

        int carry = 0;
        while (!l1_num.empty() || !l2_num.empty())
        {
            int num1 = 0, num2 = 0;
            if (!l1_num.empty())
            {
                num1 = l1_num.top();
                l1_num.pop();
            }
            if (!l2_num.empty())
            {
                num2 = l2_num.top();
                l2_num.pop();
            }

            int sum = num1 + num2 + carry;
            carry = sum / 10;
            sum = sum % 10;

            prev = node;
            node = new ListNode(sum);
            node->next = prev;
        }
        if (carry>0)
        {
            prev = node;
            node = new ListNode(carry);
            node->next = prev;
        }
        return node;
    }
};

// 3. Longest Substring Without Repeating Characters
class LongestSubstringWithoutRepeating {
public:
    int LengthOfSubstring(string s)
    {
        if (s.size() == 0) return 0;

        map<char, int> char_set;
        char_set.insert(pair<char, int>(s.at(0), 0));
        // greedy algo?
        int index = 1, max_len = 1, result = 1;
        while (index<s.size())
        {
            char curr = s.at(index);
            auto it = char_set.find(curr);
            if (it == char_set.end())
            {
                char_set[curr] = index;
                if (++max_len>result)
                    result = max_len;

                index++;
            }
            else
            {
                max_len = 0;
                index = it->second + 1;
                char_set.clear();
            }
        }
        return result;
    }
};

/*
4. Median of Two Sorted Arrays
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
e.g:
nums1 = [1, 3]
nums2 = [2]
The median is 2.0
nums1 = [1, 2]
nums2 = [3, 4]
The median is (2 + 3)/2 = 2.5
*/
class MedianofTwoSortedArray 
{
public:
    double Find(vector<int>& nums1, vector<int>& nums2)
    {
        if (nums1.size() == 0 && nums2.size() == 0) return 0;

        bool is_even = (nums1.size() + nums2.size()) % 2 == 0;
        int mid_size = (nums1.size() + nums2.size()) / 2 + 1;

        int mid = 0, mid_pre = 0;
        int id1 = 0, id2 = 0;
        for (int index = 0; index<mid_size; index++)
        {
            mid_pre = mid;
            if (id1 >= nums1.size()) mid = nums2[id2++];
            else if (id2 >= nums2.size()) mid = nums1[id1++];
            else if (nums1[id1]<nums2[id2]) mid = nums1[id1++];
            else mid = nums2[id2++];
        }

        return is_even ? (mid + mid_pre) / 2.0 : mid;
    }
};

// 5. Longest Palindromic Substring 
class LongestPalindrome
{
public:
    string BrutalWay(string s)
    {
        if (s.size()<2) return s;
        int max_start = 0, max_len = 0;
        for (int i = 1; i<s.size(); i++)
        {
            function<void(int, int)> check_functor = [&](int start, int end)
            {
                while (start >= 0 && end<s.size() && s.at(start) == s.at(end))
                {
                    start--;
                    end++;
                }
                if (end - start - 1>max_len)
                {
                    max_len = end - start - 1;
                    max_start = start + 1;
                }
            };
            // check for odd
            check_functor(i - 1, i + 1);
            // check for even
            check_functor(i - 1, i);
        }

        return s.substr(max_start, max_len);
    }

};

/*
7. Reverse Integer 
x = 123, return 321, x = -123, return -321
The input is assumed to be a 32-bit signed integer. 
Your function should return 0 when the reversed integer overflows.
*/
class ReverseInteger
{
public:
    int Solution(int x) 
    {
        bool neg = x<0;
        if (neg) x = -x;

        int res = 0;
        while (x>0)
        {
            int digit = x % 10;
            if ((INT_MAX - digit) / 10 < res) return 0;
            res = res * 10 + digit;
            x /= 10;
        }
        return neg ? -res : res;
    }
};

// 8. String to Integer (atoi)
class MyAtoiSolution 
{
public:
    int myAtoi(string str) 
    {
        int flag = 1;
        int res = 0;
        int start = str.find_first_not_of(" ");
        for (int i = start; i<str.size(); i++)
        {
            char val = str[i];
            int digit = val - '0';
            if (digit >= 0 && digit <= 9)
            {
                if (res>INT_MAX / 10 || (res == INT_MAX / 10 && digit>INT_MAX % 10))
                {
                    return flag>0 ? INT_MAX : INT_MIN;
                }
                res = res * 10 + digit;
            }
            else if (i == start && val == '-') flag = -1;
            else if (i == start && val == '+') continue;
            else break;
        }

        return res*flag;
    }
};

/*
9. Palindrome Number
Determine whether an integer is a palindrome. Do this without extra space.
*/
class PalindromeNumber
{
public:
    bool Check(int x)
    {
        if (x<0 || (x != 0 && x % 10 == 0)) return false;

        int sum = 0;

        while (x>sum)
        {
            sum = sum * 10 + x % 10;
            x = x / 10;
        }
        // even or odd
        return (x == sum || x == sum / 10);
    }
};

/*
10. Regular Expression Matching
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
isMatch("ab", ".*") �� true
isMatch("aab", "c*a*b") �� true
*/

// not correct
class SimpleRegularExpression
{
public:
    class NFAdg
    {
        enum color
        {
            white = 0,
            gray = 1,
            black = 2
        };
    public:
        NFAdg(size_t size)
        {
            SetVertex(size);
        }
        NFAdg() = default;
        ~NFAdg() = default;

        void SetVertex(size_t size)
        {
            for (int i = 0; i<size; i++)
            {
                States_.push_back(set<int>());
            }

            int prev_marked_size = Marked_.size();
            Marked_.resize(size, white);
            if (prev_marked_size > 0) ResetVisit();
        }

        void AddEdge(int i, int j)
        {
            States_[i].insert(j);    
        }

        void RemoveEdge(int i, int j)
        {
            States_[i].erase(j);
        }

        void ResetVisit()
        {
            for (int i = 0; i < Marked_.size(); i++)
                Marked_[i] = white;
        }

        void DFS(int i,vector<int> &visit)
        {
            if (Marked_[i] == white)
            {
                Marked_[i] = gray;
                visit.push_back(i);
                for (auto adj : States_[i]) DFS(adj,visit);
                Marked_[i] = black;
            }
        }

    private:
        vector<set<int>> States_;
        vector<color> Marked_;
    };

    class RegExpNFA
    {
    public:
        RegExpNFA() = default;
        ~RegExpNFA() = default;

        RegExpNFA(const string &pat) { SetPattern(pat); }

        void SetPattern(const string &pattern)
        {
            if (pattern.size() == 0) return;

            Pattern_ = pattern;
            if (Pattern_[0] != '(') Pattern_.insert(0, "(");
            if (Pattern_[Pattern_.size() - 1] != ')') Pattern_.append(")");

            size_t p_len = Pattern_.size();
            NFAdg_.SetVertex(p_len +1);

            stack<int> ops;
            for (int i = 0; i<p_len; i++)
            {
                int lp_id = i;
                if (Pattern_[i] == '(' || Pattern_[i] == '|') ops.push(i);
                else if (Pattern_[i] == ')')
                {
                    int or_id = ops.top();
                    ops.pop();
                    if (Pattern_[or_id] == '|')
                    {
                        // lp->(....|<-or...)<-i
                        lp_id = ops.top();
                        ops.pop();
                        NFAdg_.AddEdge(lp_id, or_id + 1);
                        NFAdg_.AddEdge(or_id, i);
                    }
                    else lp_id = or_id;
                }
                // two cases: lp->(...)*<-i+1  or  a* 
                if (i<p_len -1 && Pattern_[i + 1] == '*')
                {
                    NFAdg_.AddEdge(lp_id, i + 1);
                    NFAdg_.AddEdge(i + 1, lp_id);
                }
                if (Pattern_[i] == '*' || Pattern_[i] == '(' || Pattern_[i] == ')') 
                    NFAdg_.AddEdge(i, i + 1);
            }
        }

        bool Match(const string &str)
        {
            vector<int> candidates;
            NFAdg_.DFS(0, candidates);
            size_t p_len = Pattern_.size();

            for (int i = 0; i<str.size(); i++)
            {
                vector<int> match;
                for (int v : candidates)
                {
                    if (v < p_len && (Pattern_[v] == str[i] || Pattern_[v] == '.'))
                        match.push_back(v + 1);
                }

                NFAdg_.ResetVisit();
                candidates.clear();
                for (auto v : match)
                {
                    NFAdg_.DFS(v, candidates);
                }
            }

            for (int v : candidates) if (v == p_len) return true;
            return false;
        }

    private:
        NFAdg NFAdg_;
        string Pattern_;
    };

    bool isMatch(string s, string p)
    {
        if (p.size() == 0) return s.size() == 0;

        RegExpNFA nfa(p);
        return nfa.Match(s);
    }
};

/*
13. Roman to Integer
*/
class RomanToInt 
{
public:
    int Convert(string s)
    {
        if (s.size() == 0) return 0;

        unordered_map<char, int> digit_map = { { 'I' , 1 },{ 'V' , 5 },{ 'X' , 10 },
        { 'L' , 50 },{ 'C' , 100 },{ 'D' , 500 },{ 'M' , 1000 } };

        int result = digit_map[s[s.size() - 1]];
        int prev = result;
        for (int i = s.size() - 2; i >= 0; i--)
        {
            int digit = digit_map[s[i]];
            if (digit < prev) result -= digit;
            else result += digit;
            prev = digit;
        }

        return result;
    }
};

/*
14. Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
*/
class LongestCommonPrefix
{
public:
    string Solution(vector<string>& strs)
    {
        if (strs.size() == 0) return "";

        int len = strs[0].size();
        int index = 0;
        for (; index<len; index++)
        {
            int c = strs[0][index];
            bool unmatched = false;
            for (auto &str : strs)
            {
                if (index >= str.size())
                {
                    unmatched = true; break;
                }

                if (str[index] != c)
                {
                    unmatched = true; break;
                }
            }
            if (unmatched) break;
        }
        
        return strs[0].substr(0, index);
    }
};


/*
15. 3Sum
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero.
*/
class ThreeSum 
{
public:
    vector<vector<int>> Solution(vector<int>& nums)
    {
        vector<vector<int>> result;
        if (nums.size() < 3) return result;

        sort(nums.begin(), nums.end());
        for (int i = 0; i<nums.size(); i++)
        {
            TwoSum(nums[i], i + 1, nums);
        }

        result = vector<vector<int>>(answer_.begin(), answer_.end());
        return result;
    }

private:
    void TwoSum(int curr_num, int start, vector<int> &nums)
    {
        int begin = start;
        int end = nums.size() - 1;
        while (begin<end)
        {
            int sum = nums[begin] + nums[end];
            if (sum == -curr_num)
            {
                answer_.insert({ curr_num,nums[begin],nums[end] });
                begin++; end--;
            }
            else if (sum < -curr_num) begin++;
            else end--;
        }
    }
    // filter out duplication
    set<vector<int>> answer_;
};

/*
16. 3Sum Closest
For example, given array S = {-1 2 1 -4}, and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
*/
class ThreeSumClosest 
{
public:
    int Solution(vector<int>& nums, int target)
    {
        int closest_gap = INT_MAX;
        if (nums.size()<3) return closest_gap;

        sort(nums.begin(), nums.end());

        for (int i = 0; i<nums.size(); i++)
        {
            int begin = i + 1;
            int end = nums.size() - 1;
            while (begin<end)
            {
                int sum = nums[i] + nums[begin] + nums[end];
                int gap = target - sum;
                if (closest_gap == INT_MAX || abs(gap)<abs(closest_gap)) closest_gap = gap;

                if (gap == 0) return target;
                else if (gap>0) begin++;
                else end--;
            }
        }
        return target - closest_gap;
    }
};

/*
20. Valid Parentheses
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
*/
class ValidParentheses
{
public:
    bool Check(string s)
    {
        unordered_map<char, char> par_pair;
        par_pair['('] = ')';
        par_pair['{'] = '}';
        par_pair['['] = ']';
        stack<char> par;
        for (char c : s)
        {
            if (par_pair.find(c) != par_pair.end()) par.push(par_pair[c]);
            else if (!par.empty() && par.top() == c) par.pop();
            else return false;
        }
        return par.empty();
    }
};

/*
21. Merge Two Sorted Lists
Merge two sorted linked lists and return it as a new list. 
The new list should be made by splicing together the nodes of the first two lists.
*/
class Solution
{
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2)
    {
        ListNode *result = NULL, *prev = NULL;
        while (l1 != NULL || l2 != NULL)
        {
            ListNode *node = NULL;
            if (l1 == NULL)
            {
                node = l2;
                l2 = l2->next;
            }
            else if (l2 == NULL)
            {
                node = l1;
                l1 = l1->next;
            }
            else if (l1->val<l2->val)
            {
                node = l1;
                l1 = l1->next;
            }
            else
            {
                node = l2;
                l2 = l2->next;
            }
            if (prev == NULL) result = prev = node;
            else { prev->next = node; prev = node; }
        }
        return result;
    }
};

/*
28. Implement strStr()
Returns the index of the first occurrence of needle in haystack or -1(not found).
*/
class MyStrStr 
{
public:
    int strstr(const string &haystack, const string &needle)
    {
        if (needle.size() == 0) return 0;

        int last = haystack.size() - needle.size() + 1;
        for (int i = 0; i<last; i++)
        {
            int j = 0;
            int temp = i;
            for (; j<needle.size(); j++)
            {
                if (haystack[temp++] != needle[j]) break;
            }
            if (j == needle.size()) return i;
        }
        return -1;
    }

    int sunday(const string &haystack, const string &needle)
    {
        // sunday version
        if (needle.size() == 0) return 0;

        // preproccess pattern
        vector<int> char_set(256, -1);
        for (int i = 0; i<needle.size(); i++)
        {
            char_set[needle[i] - 'a'] = i;
        }

        int last_check = haystack.size() - needle.size();
        for (int i = 0; i <= last_check;)
        {
            int j = 0;
            int temp = i;
            for (; j<needle.size(); j++)
            {
                if (haystack[temp++] != needle[j])
                {
                    // mismatch
                    int index = i + needle.size();
                    if (index<haystack.size())
                    {
                        int val = char_set[haystack[index] - 'a'];
                        if (val == -1) i = index+1;
                        else i = index - val;
                        break;
                    }
                    else return -1;
                }
            }
            if (j == needle.size()) return i;
        }
        return -1;
    }
};

/*
36. Valid Sudoku
The Sudoku board could be partially filled, 
where empty cells are filled with the character '.'.
*/
class ValidSudoku 
{
public:
    bool IsValid(vector<vector<char>>& board) 
    {
        bool row[9][9] = { false };
        bool col[9][9] = { false };
        bool sub[9][9] = { false };

        for (int i = 0; i<board.size(); i++)
        {
            for (int j = 0; j<board[i].size(); j++)
            {
                if (board[i][j] != '.')
                {
                    int num = board[i][j] - '0' - 1;
                    int sub_id = i / 3 * 3 + j / 3;
                    // check row
                    if (row[i][num] ||
                        col[j][num] ||
                        sub[sub_id][num]) return false;
                    else row[i][num] = col[j][num] = sub[sub_id][num] = true;
                }
            }
        }
        return true;

    }
};

/*
37. Sudoku Solver
*/
class SudokuSolver 
{
public:
    void Solve(vector<vector<char>>& board)
    {
        Helper(board);
    }

private:
    bool Helper(vector<vector<char>>& board)
    {
        for (int i = 0; i<board.size(); i++)
        {
            for (int j = 0; j<board[i].size(); j++)
            {
                if (board[i][j] == '.')
                {
                    for (int k = 0; k<9; k++)
                    {
                        char c = '1' + k;
                        if (IsValid(board, c, i, j))
                        {
                            board[i][j] = c;
                            // try this solution
                            if (Helper(board))
                                return true;
                            else
                                board[i][j] = '.'; // backtracking   
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    bool IsValid(vector<vector<char>>& board, char c, int i, int j)
    {
        for (int k = 0; k<board[i].size(); k++)
        {
            // check row
            if (board[i][k] == c) return false;
            // check column
            if (board[k][j] == c) return false;
            // check sub-box
            int sub_r = i / 3 * 3 + k / 3; // 000111...
            int sub_c = j / 3 * 3 + k % 3; // 012012...
            if (board[sub_r][sub_c] == c) return false;
        }
        return true;
    }
};

/*
44. Wildcard Matching
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
*/
class WildCardMatch
{
public:
    bool IsMatch(string s, string p)
    {
        int p_index = 0;
        int s_index = 0;
        int star_index = -1;
        int star_matched = 0;
        while (s_index<s.size())
        {
            if (p_index<p.size() && (p.at(p_index) == s.at(s_index) || p.at(p_index) == '?'))
            {
                p_index++;
                s_index++;
            }
            else if (p_index<p.size() && p.at(p_index) == '*')
            {
                // s_index meets *
                star_index = ++p_index;
                star_matched = s_index;
            }
            else if (star_index != -1)
            {
                // it's under * mode
                p_index = star_index;
                s_index = ++star_matched;
            }
            else return false;
        }

        while (p_index<p.size() && p.at(p_index) == '*') p_index++;
        return p_index == p.size();
    }
};

/*
48. Rotate Image
*/
class RotateImage 
{
public:
    void MirrorFlip(vector<vector<int>>& matrix)
    {
        int n = matrix.size();
        // diagonal ��/��
        for (int i = 0; i<n - 1; i++)
        {
            for (int j = 0; j<n - i; j++)
            {
                swap(matrix[i][j], matrix[n - j - 1][n - i - 1]);
            }
        }

        // horizal '-'
        for (int i = 0; i<n / 2; i++)
        {
            for (int j = 0; j<n; j++)
            {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
    }

    void DirectLyMove(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        for (int layer = 0; layer<n / 2; layer++)
        {
            int end = n - layer - 1;
            // only move n-1
            for (int i = layer; i<end; i++)
            {
                int offset = n - i - 1;
                // save top
                int temp = matrix[layer][i];
                // move left to top
                matrix[layer][i] = matrix[offset][layer];
                // move bottom to left
                matrix[offset][layer] = matrix[end][offset];
                // move right to bottom
                matrix[end][offset] = matrix[i][end];
                // move top to right
                matrix[i][end] = temp;
            }
        }
    }
};

/*
51 N Queens return solutions
*/
class NQueens 
{
public:
    vector<vector<string>> Solve(int n)
    {
        if (n == 0) return result_;

        vector<string> one_sln(n, string(n, '.'));
        DFS(one_sln, 0);
        return result_;
    }

private:
    void DFS(vector<string> &one_sln, int row)
    {
        if (row == one_sln.size())
        {
            result_.push_back(one_sln);
        }

        // goes row by row, full permutate in each row
        for (int col = 0; col< one_sln.size(); col++)
        {
            // check each col |, diagonal '/' '\'
            if(IsValid(one_sln,row,col))
            {
                one_sln[row][col] = 'Q';
                DFS(one_sln, row + 1);
                //backtracking, remove previous step
                one_sln[row][col] = '.';
            }
        }
    }

    bool IsValid(vector<string> &one_sln, int row, int col)
    {
        // check columns
        for (int i = 0; i<row; i++)
        {
            if (one_sln[i][col] == 'Q') return false;
        }

        // check diagonal '\'
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
        {
            if (one_sln[i][j] == 'Q') return false;
        }

        // check  diagonal '/'
        for (int i = row - 1, j = col + 1; i >= 0 && j <= one_sln.size(); i--, j++)
        {
            if (one_sln[i][j] == 'Q') return false;
        }
        return true;
    }

    vector<vector<string>> result_;
};

/*
52 N Queens return solution count
*/
class NQueensII 
{
public:
    int Solve(int n)
    {
        if (0 == n) return 0;
        BoardSize_ = n;
        Result_ = 0;
        Column_ = vector<int>(n, 0);
        MainDiagonal_ = vector<int>(2 * n - 1, 0);
        CounterDiagonal_ = vector<int>(2 * n - 1, 0);
        DFS(0);
        return Result_;
    }

private:
    void DFS(int row)
    {
        if (row == BoardSize_)
        {
            Result_++;
            return;
        }
        // goes row by row, full permute in each row
        for (int col = 0; col<BoardSize_; col++)
        {
            if (CheckFlag(row, col))
            {
                SetFlag(row, col, 1);
                DFS(row + 1);
                // backtracking 
                SetFlag(row, col, 0);
            }
        }
    }

    bool CheckFlag(int row, int col)
    {
        return (Column_[col] == 0 &&
            MainDiagonal_[BoardSize_ - 1 - row + col] == 0 &&
            CounterDiagonal_[row + col] == 0);
    }

    void SetFlag(int row, int col, int value)
    {
        Column_[col] = value;
        MainDiagonal_[BoardSize_ - 1 - row + col] = value;
        CounterDiagonal_[row + col] = value;
    }

    int BoardSize_;
    int Result_;
    // pruning by line '|' '\' '/'
    vector<int> Column_;
    vector<int> MainDiagonal_;
    vector<int> CounterDiagonal_;
};

/*
54. Spiral Matrix
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
*/
class SpiralMatrix
{
public:
    vector<int> SpiralOrder(vector<vector<int>> &matrix)
    {
        vector<int> result;
        if (matrix.size() == 0) return result;

        int x_begin = 0, x_end = matrix[0].size() - 1;
        int y_begin = 0, y_end = matrix.size() - 1;
        while (true)
        {
            // copy top line, left to right
            for (int i = x_begin; i <= x_end; i++) result.push_back(matrix[y_begin][i]);
            if (++y_begin > y_end) break;

            // copy rigt line, top to bottom
            for (int i = y_begin; i <= y_end; i++) result.push_back(matrix[i][x_end]);
            if (--x_end < x_begin) break;

            // copy bottom line, right to left
            for (int i = x_end; i >= x_begin; i--) result.push_back(matrix[y_end][i]);
            if (--y_end < y_begin) break;

            // copy left line, bottom to top
            for (int i = y_end; i >= y_begin; i--) result.push_back(matrix[i][x_begin]);
            if (++x_begin > x_end) break;
        }
        return result;
    }
};

/*
55 Jump Game
Given an array of non-negative integers, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.
*/
class JumpGame {
public:
    bool CanJump(vector<int>& nums)
    {
        // greedy to check if it can reach 
        int index = 0;
        int max_reachable = 0;
        for (; index<nums.size(); index++)
        {
            if (max_reachable < index) return false;
            max_reachable = max(max_reachable, nums[index] + index);
            if (max_reachable >= nums.size() - 1) return true;
        }
        return index == nums.size() - 1;
    }

};

/*
75. Sort Colors
Given an array with n objects colored red, white or blue, 
sort them so that objects of the same color are adjacent,
with the colors in the order red, white and blue. Here, we will use the integers 0, 1, and 2 
to represent the color red, white, and blue respectively.
*/
class SortColors
{
    void Sort(vector<int>& nums)
    {
        int begin = 0;
        int index = 0;
        int end = nums.size() - 1;
        while (index <= end)
        {
            switch (nums[index])
            {
            case 0: // red
                swap(nums[index++], nums[begin++]);
                break;
            case 2: // blue
                swap(nums[index], nums[end--]);
                break;
            default:
                index++;
                break;
            }
        }
    }
};

/*
88. Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
You may assume that nums1 has enough space (size that is greater or equal to m + n) 
to hold additional elements from nums2. 
The number of elements initialized in nums1 and nums2 are m and n respectively.
*/
class MergeSortedArray 
{
public:
    void Merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int index = m + n - 1;
        int id1 = m - 1, id2 = n - 1;

        while (id1 >= 0 && id2 >= 0)
        {
            if (nums1[id1]>nums2[id2]) nums1[index--] = nums1[id1--];
            else nums1[index--] = nums2[id2--];
        }
        // only process id2 if remaining, id1 is in place
        while (id2 >= 0) nums1[index--] = nums2[id2--];
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
121. Best Time to Buy and Sell Stock  
If you were only permitted to complete at most one transaction 
(ie, buy one and sell one share of the stock)
Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5
max. difference = 6-1 = 5 (not 7-1 = 6��
*/
class BestTimeBuySellStock
{
public:
    int MaxProfit(vector<int>& prices) 
    {
        int max = 0;
        int buy = INT_MAX;
        for (int i = 0; i<prices.size(); i++)
        {
            int profit = prices[i] - buy;
            if (profit > max) max = profit;
            if (prices[i] < buy) buy = prices[i];
        }
        return max;
    }
};

/*
122. Best Time to Buy and Sell Stock II
You may complete as many transactions as you like (ie, buy one and sell of the stock multiple times). 
However, you may not engage in multiple transactions at the same time 
(ie, you must sell the stock before you buy again).
*/
class BestTimeBuySellStockII {
public:
    int maxProfit(vector<int>& prices)
    {
        int sum = 0;
        int buy = INT_MAX;
        int sell = INT_MIN;
        int previous = INT_MAX;
        for (int i = 0; i<prices.size(); i++)
        {
            if (prices[i] < previous)
            {
                // commit preivous buy/sell
                if (previous > buy)
                {
                    sum += (previous - buy);
                }
                buy = prices[i];
            }
            previous = prices[i];
        }
        if (previous>buy) sum += (previous - buy);
        return sum;
    }
};

class BestTimeSellStockIII
{
public:    
    int maxProfit(vector<int> &prices)
    {
        
    }
};

/*
125. Valid Palindrome
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.
*/
class ValidPalindrome {
public:
    bool Check(string s)
    {
        if (s.size() == 0) return true;

        int begin = 0;
        int end = s.size() - 1;
        while (begin<end)
        {
            if (!isalnum(s[begin]))
            {
                begin++;
                continue;
            }

            if (!isalnum(s[end]))
            {
                end--;
                continue;
            }
            if (tolower(s[begin]) != tolower(s[end])) return false;

            begin++;
            end--;
        }
        return true;
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

class WordLadderSolution 
{
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
138. Copy List with Random Pointer
A linked list is given such that each node contains an additional random pointer 
which could point to any node in the list or null.
Return a deep copy of the list.
*/
struct RandomListNode
{
    int label;
    RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};
class CopyRandomList
{
public:
    RandomListNode *CopyList(RandomListNode *head)
    {
        if (head == NULL) return NULL;
        RandomListNode *curr = head;
        RandomListNode *new_node = NULL;
        RandomListNode *next_node = NULL;
        RandomListNode *new_head = NULL;
        // added new node after old one 
        // old1->new1->old2->new2
        while (curr)
        {
            new_node = new RandomListNode(curr->label);
            next_node = curr->next;
            curr->next = new_node;
            new_node->next = next_node;
            curr = next_node;
        }
        // modify random for new old
        curr = head;
        while (curr)
        {
            new_node = curr->next;
            new_node->random = curr->random == NULL ? NULL : curr->random->next;

            curr = new_node->next;
        }
        // split old/new nodes
        curr = head;
        while (curr)
        {
            new_node = curr->next;
            if (new_head == NULL)
                new_head = new_node;

            curr->next = new_node->next;
            curr = new_node->next;
            new_node->next = curr == NULL ? NULL : curr->next;
        }
        return new_head;
    }
};

/*
141. Linked List Cycle
Given a linked list, determine if it has a cycle in it.
*/
class LinkedListCycle 
{
public:
    bool hasCycle(ListNode *head) 
    {
        if (head == NULL) return false;

        ListNode *slow = head;
        ListNode *fast = head;

        while (fast != NULL && fast->next != NULL)
        {
            slow = slow->next; // move 1 step
            fast = fast->next->next; // move 2 steps
            if (fast == slow) return true;
        }
        return false;
    }
};

/*
142. Linked List Cycle II
*/
class LinkedListCycleII 
{
public:
    ListNode *detectCycle(ListNode *head)
    {
        if (head == NULL) return NULL;

        ListNode *slow = head;
        ListNode *fast = head;
        // find the meeting point
        ListNode *meet = NULL;
        while (fast != NULL && fast->next != NULL)
        {
            slow = slow->next;
            fast = fast->next->next;
            if (fast == slow)
            {
                meet = fast;
                break;
            }
        }
        if (meet == NULL) return NULL;

        slow = head;
        while (slow != fast)
        {
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};

/*
165 Compare Version Numbers
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
0.1 < 1.1 < 1.2 < 13.37
*/
class CompareVersionNumbers 
{
public:
    int Comparee(string version1, string version2)
    {
        stringstream s1(version1);
        stringstream s2(version2);

        while (true)
        {
            string v1, v2;

            istream &p1 = getline(s1, v1, '.');
            istream &p2 = getline(s2, v2, '.');

            if (!p1 && !p2) break;

            int n1 = atoi(v1.c_str());
            int n2 = atoi(v2.c_str());

            if (n1 == n2) continue;
            return n1>n2 ? 1 : -1;
        }

        return 0;
    }
};

/*
171. Excel Sheet Column Number
*/
class ExcelSheetColumnNumber
{
    int titleToNumber(string s)
    {
        int number = 0;
        for (auto val : s)
        {
            char c_digit = toupper(val);
            int n_digit = c_digit - 'A' + 1;
            number = number * 26 + n_digit;
        }
        return number;
    }
};

/*
191. Number of 1 Bits
*/
class NumberOfBits 
{
public:
    int hammingWeight(uint32_t n) 
    {
        int i = 0;
        while (n>0)
        {
            i++;
            n &= (n - 1);
        }
        return i;
    }
};

/*
200. Number of Islands
*/
class NumberOfIslands
{
public:
    int CountIslands(vector<vector<char>>& grid)
    {
        if (grid.size() == 0) return 0;

        int nums = 0;
        int x_len = grid.size();
        int y_len = grid[0].size();

        for (int i = 0; i<x_len; i++)
        {
            for (int j = 0; j<y_len; j++)
            {
                if (grid[i][j] == '1')
                {
                    BFS(i, j, grid);
                    nums++;
                }
            }
        }
        return nums;
    }

    void BFS(int i, int j, vector<vector<char>>&grid)
    {
        struct point
        {
            point(int i, int j) :
                x(i), y(j) {}
            ~point() = default;
            int x;
            int y;
        };
        int x_max = grid.size();
        int y_max = grid[0].size();
        queue<point> path;
        path.push(point(i, j));
        while (!path.empty())
        {
            point xy = path.front();
            path.pop();
            i = xy.x;
            j = xy.y;
            grid[i][j] = '0';
            // check xy's adjacent
            function<void(int, int)> processor = [&](int x, int y)
            { if (grid[x][y] == '1') {
                path.push(point(x, y)); grid[x][y] = '2';
            }
            };
            if (i>0) processor(i - 1, j);
            if (j>0) processor(i, j - 1);
            if (i < x_max - 1) processor(i + 1, j);
            if (j < y_max - 1) processor(i, j + 1);
        }
    }
};

/*
232. Implement Queue using Stacks
*/
class MyQueueByStack 
{
public:
    MyQueueByStack() = default;
    ~MyQueueByStack() = default;

    void push(int x) {
        stack<int> temp;
        // reverse data
        while (!Data_.empty())
        {
            temp.push(Data_.top());
            Data_.pop();
        }
        // push back 
        Data_.push(x);
        while (!temp.empty())
        {
            Data_.push(temp.top());
            temp.pop();
        }
    }

    int pop() {
        int ret = 0;
        ret = Data_.top();
        Data_.pop();
        return ret;
    }

    int peek() { return Data_.top();}
    bool empty() { return Data_.empty();}

private:
    stack<int> Data_;
};

/*
237. Delete Node in a Linked List
*/
class DeleteNodeInLinkedList
{
public:
    void deleteNode(ListNode* node)
    {
        if (node == NULL) return;
        ListNode *prev = node;
        while (node && node->next)
        {
            node->val = node->next->val;
            prev = node;
            node = node->next;
        }
        prev->next = NULL;
        delete node;
    }
};

/*
258. Add Digits
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
e.g:
Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
*/
class AddDigits 
{
public:
    int add(int num) 
    {

        while (num>9)
        {
            num = num / 10 + num % 10;
        }
        return num;
    }

    // digit of number added == number%9
    int add2(int num) {

        int res = num % 9;
        if (res == 0 && num != 0) res = 9;

        return res;
    }
};

/*
419. Battleships in a Board
*/
class BattleshipsInBoard
{
public:
    int CountBS(vector<vector<char>>& board)
    {
        int count = 0;
        for (int i = 0; i<board.size(); i++)
        {
            for (int j = 0; j<board[i].size(); j++)
            {
                if (board[i][j] != '.')
                {
                    if (i == 0 || board[i - 1][j] == '.')
                    {
                        if (j == 0 || board[i][j - 1] == '.')
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
