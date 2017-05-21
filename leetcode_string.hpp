#ifndef LEETCODE_STRING_HPP
#define LEETCODE_STRING_HPP

#include "leetcode.hpp"

/**************************************************
                    String
***************************************************/

using namespace std;

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
                if (++max_len>result) result = max_len;

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
10. Regular Expression Matching
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
isMatch("ab", ".*") ¡ú true
isMatch("aab", "c*a*b") ¡ú true
*/
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

        void DFS(int i, vector<int> &visit)
        {
            if (Marked_[i] == white)
            {
                Marked_[i] = gray;
                visit.push_back(i);
                for (auto adj : States_[i]) DFS(adj, visit);
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
            NFAdg_.SetVertex(p_len + 1);

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
                if (i<p_len - 1 && Pattern_[i + 1] == '*')
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
                        if (val == -1) i = index + 1;
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
387. First Unique Character in a String
Given a string, find the first non-repeating character in it and return it's index.
If it doesn't exist, return -1.

Examples:
s = "leetcode"
return 0.
*/
class FirstUniqueCharacter {
public:
    int firstUniqChar(string s) {
        vector<int> counter(256, 0);
        for (char c : s) counter[c]++;
        for (int i = 0; i<s.size(); i++)
        {
            if (counter[s[i]] == 1) return i;
        }
        return -1;
    }
};

#endif