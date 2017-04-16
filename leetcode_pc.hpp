#ifndef LEETCODE_PC_HPP
#define LEETCODE_PC_HPP

#include "leetcode.hpp"
using namespace std;

/**************************************************
                Permutation
***************************************************/

/*
31. Next Permutation
*/
void NextPermutation(vector<int>& nums)
{
    if (nums.size() < 2) return;

    int partition_index = nums.size() - 1;
    int prev = nums[partition_index--];
    // find the partition number (first non creasing index from right to left)
    while (partition_index>0)
    {
        if (nums[partition_index] < prev) break;
        else
        {
            prev = nums[partition_index];
            partition_index--;
        }
    }
    // find the change number (smallest index greater than partition number from right to left)
    int change_index = nums.size() - 1;
    while (change_index>partition_index)
    {
        if (nums[change_index] > nums[partition_index]) break;
        else change_index--;
    }
    // reverse numuber after partition index
    int left = partition_index, right = nums.size() - 1;
    if (change_index != partition_index)
    {
        swap(nums[change_index], nums[partition_index]);
        // change == partion suggests nums is the largest permutation,reverse whole
        left++;
    }

    for (; left<right; left++, right--)
    {
        swap(nums[left], nums[right]);
    }
}

/*
46. Permutations
Given a collection of distinct numbers, return all possible permutations.
*/
class Permutation
{
public:
    vector<vector<int>> Permute(vector<int>& nums)
    {
        vector<vector<int>> result;
        if (nums.size() == 0) return result;

        FullPermute(result, nums, 0);
        return result;
    }

private:
    void FullPermute(vector<vector<int>> &result, vector<int>& nums, int start)
    {
        if (start == nums.size())
        {
            result.push_back(nums);
            return;
        }

        for (int i = start; i<nums.size(); i++)
        {
            // P[i] = A[i] + P[i+1,N]
            swap(nums[start], nums[i]);
            FullPermute(result, nums, start + 1);
            swap(nums[start], nums[i]);
        }
    }
};

/*
47. Permutations II
Given a collection of numbers that might contain duplicates, return all possible unique permutations.
*/
class PermutationII
{
public:
    vector<vector<int>> PermuteUnique(vector<int>& nums)
    {
        vector<vector<int>> result;
        DFS(0, nums, result);
        return result;
    }

private:
    void DFS(int start, vector<int> &nums, vector<vector<int>> &result)
    {
        if (start == nums.size())
        {
            result.push_back(nums);
            return;
        }

        for (int i = start; i<nums.size(); i++)
        {
            if (IsUnique(nums, start, i))
            {
                swap(nums[i], nums[start]);
                DFS(start + 1, nums, result);
                // fallback
                swap(nums[i], nums[start]);
            }
        }
    }

    bool IsUnique(vector<int> &nums, int start, int i)
    {
        bool is_unique = true;
        for (int j = start; j<i; j++)
        {
            if (nums[j] == nums[i])
            {
                is_unique = false;
                break;
            }
        }
        return is_unique;
    }
};

/**************************************************
                Combination
***************************************************/
/*
77. Combinations
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
*/
class Combinations
{
public:
    vector<vector<int>> CombineRecursion(int n, int k)
    {
        vector<vector<int>> result;
        vector<int> group;
        int start = 0;
        dfs(n, k, 1, group, result);
        return result;
    }

    vector<vector<int>> CombineIteration(int n, int k)
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

    void dfs(int n, int k, int start, vector<int> &group, vector<vector<int>> &result)
    {
        if (0 == k)
        {
            result.push_back(group);
        }
        else
        {
            for (int i = start; i <= n; i++)
            {
                group.push_back(i);
                dfs(n, k - 1, i + 1, group, result);
                group.pop_back();
            }
        }
    }
};

/*
78. Subsets
Given a set of distinct integers, nums, return all possible subsets.
*/
class Subsets
{
public:
    vector<vector<int>> CountRecursive(vector<int>& nums)
    {
        vector<vector<int>> result;
        vector<int> one_sln;
        DFS(result, one_sln, nums, 0);
        return result;
    }

    vector<vector<int>> CountBitManipulation(vector<int>& nums)
    {
        //sort(nums.begin(), nums.end());
        int num_subset = pow(2, nums.size());
        vector<vector<int> > res(num_subset, vector<int>());
        for (int i = 0; i < nums.size(); i++)
            for (int j = 0; j < num_subset; j++)
                if ((j >> i) & 1)
                    res[j].push_back(nums[i]);
        return res;
    }

private:
    void DFS(vector<vector<int>>&result, vector<int>&one_sln, vector<int>& nums, int start)
    {
        result.push_back(one_sln);
        for (int i = start; i< nums.size(); i++)
        {
            one_sln.push_back(nums[i]);
            DFS(result, one_sln, nums, i + 1);
            one_sln.pop_back();
        }
    }
};

/*
90. Subsets II
Given a collection of integers that might contain duplicates, nums, return all possible subsets.
*/
class SubsetsII 
{
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums)
    {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;
        vector<int> one_sln;
        DFS(result, one_sln, nums, 0);
        return result;
    }

private:
    void DFS(vector<vector<int>>&result, vector<int>&one_sln, vector<int>& nums, int start)
    {
        result.push_back(one_sln);
        for (int i = start; i< nums.size(); i++)
        {
            if (i == start || nums[i] != nums[i - 1])
            {
                one_sln.push_back(nums[i]);
                DFS(result, one_sln, nums, i + 1);
                one_sln.pop_back();
            }
        }
    }
};

/*
39. Combination Sum
Given a set of numbers (without duplicates) and a target number (T), 
find all unique combinations where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of time
For example, given candidate set [2, 3, 6, 7] and target 7,
A solution set is:
[
[7],
[2, 2, 3]
]
*/
class CombinationSum
{
public:
    vector<vector<int>> Solution(vector<int>& candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        Helper(candidates, target, 0);
        return Result_;
    }

private:
    void Helper(vector<int>& cand, int target, int start)
    {
        if (target == 0)
        {
            Result_.push_back(OneGroup_);
            return;
        }

        for (int i = start; i<cand.size() && cand[i] <= target; i++)
        {
            OneGroup_.push_back(cand[i]);
            // go no further(not i+1, retry current i) until it fails to add up
            Helper(cand, target - cand[i], i);
            OneGroup_.pop_back();
        }
    }

    vector<vector<int>> Result_;
    vector<int> OneGroup_;
};

/*
Given a collection of candidate numbers (C) and a target number (T), 
find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.
For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8,
A solution set is:
[
[1, 7],
[1, 2, 5],
[2, 6],
[1, 1, 6]
]
*/
class CombinationSumII 
{
public:
    vector<vector<int>> Solution(vector<int>& candidates, int target)
    {
        sort(candidates.begin(), candidates.end());
        Helper(candidates, target, 0);
        return Result_;
    }

private:
    void Helper(vector<int>& cand, int target, int start)
    {
        if (target == 0)
        {
            Result_.push_back(OneGroup_);
            return;
        }

        for (int i = start; i<cand.size() && cand[i] <= target; i++)
        {
            if (i == start || cand[i] != cand[i - 1])
            {
                OneGroup_.push_back(cand[i]);
                Helper(cand, target - cand[i], i + 1);
                OneGroup_.pop_back();
            }
        }
    }

    vector<vector<int>> Result_;
    vector<int> OneGroup_;
};


/*
216. Combination Sum III
Find all possible combinations of k numbers that add up to a number n,
given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Example 2:
Input: k = 3, n = 9
Output:
[[1,2,6], [1,3,5], [2,3,4]]
*/
class CombinationSum3
{
public:
    vector<vector<int>> Solution(int k, int n) 
    {
        Helper(1,k,n);
        return Result_;
    }
    
private:
    void Helper(int start, int remaining, int target)
    {
        if(target == 0 && remaining==0) 
        {
            Result_.push_back(Group_);
            return;
        }
        
        for(int i=start;i<=9 && i<=target; i++)
        {
            Group_.push_back(i);
            Helper(i+1,remaining-1,target-i);
            Group_.pop_back();
        }
    }
    
    vector<vector<int>> Result_;
    vector<int> Group_;
};

#endif
