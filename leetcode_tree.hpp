#ifndef LEETCODE_TREE_HPP
#define LEETCODE_TREE_HPP

#include "leetcode.hpp"
using namespace std;

/**************************************************
                    Tree
***************************************************/

struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

/*
94, Binary Tree Inorder Traversal
Inorder(Node*root)
if root == null return;
InOrder(root->left);
visit(root);
Inorder(root->right);
*/

/*
98. Validate Binary Search Tree
left < current < right
*/
class IsValidBST
{
public:
    // based on iteration not recursion
    bool Check(TreeNode* root)
    {
        if (root == NULL) return true;

        stack<TreeNode*> visit_node;

        TreeNode *pre_node = NULL;
        while (!visit_node.empty() || root != NULL)
        {
            // push left child
            while (root != NULL)
            {
                visit_node.push(root);
                root = root->left;
            }

            if (!visit_node.empty())
            {
                root = visit_node.top();

                if (pre_node != NULL && pre_node->val >= root->val) return false;
                pre_node = root;
                visit_node.pop();
                root = root->right;
            }
        }
        return true;
    }
};

/*
102. Binary Tree Level Order Traversal
103. Binary Tree Zigzag Level Order Traversal
Given a binary tree, return the zigzag level order traversal of its nodes' values.
(ie, from left to right, then right to left for the next level and alternate between).
3
/ \
9  20
/  \
15   7
input:
[3,9,20,null,null,15,7]
output:
[3],
[20,9],
[15,7]
*/
class BinaryTreeLevelOrder
{
public:
    vector<vector<int>> LevelOrder(TreeNode* root)
    {
        vector<vector<int>> result;
        if (root == NULL) return result;

        queue<TreeNode*> visit_node;
        visit_node.push(root);
        while (!visit_node.empty())
        {
            int same_depth_num = visit_node.size();
            vector<int> depth_node;
            // consume all same depth nodes
            for (int i = 0; i<same_depth_num; i++)
            {
                TreeNode *curr = visit_node.front();
                depth_node.push_back(curr->val);

                if (curr->left != NULL) visit_node.push(curr->left);
                if (curr->right != NULL) visit_node.push(curr->right);

                visit_node.pop();
            }
            result.push_back(depth_node);
        }
        return result;
    }

    vector<vector<int>> ZigzagLevelOrder(TreeNode* root)
    {
        vector<vector<int>> result;

        if (root == NULL) return result;

        queue<TreeNode*> visit_node;
        visit_node.push(root);
        bool from_left_right = true;
        while (!visit_node.empty())
        {
            int same_depth_num = visit_node.size();
            vector<int> same_depth_node(same_depth_num);

            for (int i = 0; i<same_depth_num; i++)
            {
                TreeNode *curr = visit_node.front();
                int index = from_left_right ? i : same_depth_num - 1 - i;
                same_depth_node[index] = curr->val;

                if (curr->left != NULL) visit_node.push(curr->left);
                if (curr->right != NULL) visit_node.push(curr->right);
                visit_node.pop();
            }
            from_left_right = !from_left_right;
            result.push_back(same_depth_node);
        }
        return result;
    }
};

/*
105. Construct Binary Tree from Preorder and Inorder Traversal
*/
class PreorderInorderBuildTree {
public:
    TreeNode* BuildTree(vector<int>& preorder, vector<int>& inorder)
    {
        if (preorder.size() == 0) return NULL;

        return Helper(0, 0, inorder.size() - 1, preorder, inorder);
    }

private:
    TreeNode* Helper(int pre_index, int in_start, int in_end, vector<int>& preorder, vector<int>& inorder)
    {
        if (in_start>in_end) return NULL;

        TreeNode *root = new TreeNode(preorder[pre_index]);

        int in_index = 0;
        for (int i = in_start; i <= in_end; i++)
        {
            if (preorder[pre_index] == inorder[i])
            {
                in_index = i;
                break;
            }
        }

        root->left = Helper(pre_index + 1, in_start, in_index - 1, preorder, inorder);
        root->right = Helper(pre_index + in_index - in_start + 1, in_index + 1, in_end, preorder, inorder);

        return root;
    }
};

/*
106. Construct Binary Tree from Inorder and Postorder Traversal
*/
class PostorderInorderBuildTree
{
public:
    TreeNode* BuildTree(vector<int>& inorder, vector<int>& postorder)
    {
        if (postorder.size() == 0) return NULL;

        return Helper(postorder.size() - 1, 0, inorder.size() - 1, postorder, inorder);
    }
private:
    TreeNode* Helper(int post_index, int in_start, int in_end, vector<int>& postorder, vector<int>& inorder)
    {
        if (in_start>in_end) return NULL;

        TreeNode *root = new TreeNode(postorder[post_index]);

        int in_index = 0;
        for (int i = in_start; i <= in_end; i++)
        {
            if (postorder[post_index] == inorder[i])
            {
                in_index = i;
                break;
            }
        }

        root->left = Helper(post_index + in_index - in_end - 1, in_start, in_index - 1, postorder, inorder);
        root->right = Helper(post_index - 1, in_index + 1, in_end, postorder, inorder);

        return root;
    }
};

/*
173. Binary Search Tree Iterator
Calling next() will return the next smallest number in the BST.
Note: next() and hasNext() should run in average O(1) time and uses O(h) memory,
where h is the height of the tree.
*/
class BSTIterator {
public:
    BSTIterator(TreeNode *root)
    {
        MostLeftChild(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext()
    {
        return !VisitPath_.empty();
    }

    /** @return the next smallest number */
    int next()
    {
        // no border checking here
        TreeNode *curr_node = VisitPath_.top();
        VisitPath_.pop();
        // for next InOrder node
        MostLeftChild(curr_node->right);
        return curr_node->val;
    }

private:
    void MostLeftChild(TreeNode *curr)
    {
        while (curr != NULL)
        {
            VisitPath_.push(curr);
            curr = curr->left;
        }
    }
    stack<TreeNode*> VisitPath_;
};

/*
208. Implement Trie (Prefix Tree)
*/
class BasicTrie
{
#define SET_SIZE 26
    struct TreeNode
    {
        vector<TreeNode*> CharSet;
        bool  IsWord;
        TreeNode()
            :CharSet(SET_SIZE, NULL),
            IsWord(false) {}
    };

public:
    /** Initialize your data structure here. */
    BasicTrie()
        :Root_(new TreeNode())
    {}

    /** Inserts a word into the trie. */
    void insert(string word)
    {
        TreeNode *curr = Root_;
        for (char c : word)
        {
            if (curr->CharSet[c - 'a'] == NULL) curr->CharSet[c - 'a'] = new TreeNode();
            curr = curr->CharSet[c - 'a'];
        }
        curr->IsWord = true;
    }

    /** Returns if the word is in the trie. */
    bool search(string word)
    {
        TreeNode *last = FindNode(word);
        return last != NULL ? last->IsWord : false;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix)
    {
        return FindNode(prefix) != NULL;
    }

private:
    TreeNode *FindNode(const string &str)
    {
        TreeNode *curr = Root_;
        for (char c : str)
        {
            if (curr->CharSet[c - 'a'] == NULL) return NULL;
            curr = curr->CharSet[c - 'a'];
        }
        return curr;
    }

    TreeNode *Root_;
};

/*
235. Lowest Common Ancestor of a Binary Search Tree
*/
class LCA_BST {
public:
    TreeNode* LCA(TreeNode* root, TreeNode* p, TreeNode* q)
    {
        if (root == NULL || p == NULL || q == NULL) return NULL;

        if (root->val > p->val && root->val > q->val) return LCA(root->left, p, q);
        if (root->val < p->val && root->val < q->val) return LCA(root->right, p, q);
        else return root;
    }
};

/*
236. Lowest Common Ancestor of a Binary Tree
*/
class LCA_BT
{
public:
    TreeNode* LCA(TreeNode* root, TreeNode* p, TreeNode* q)
    {
        if (root == NULL || root == p || root == q) return root;

        // find in left/right child
        TreeNode *left = LCA(root->left, p, q);
        TreeNode *right = LCA(root->right, p, q);
        if (left != NULL && right != NULL) return root;
        else if (left != NULL) return left;
        else if (right != NULL) return right;
        else return NULL;
    }
};

/*
285 Inorder Successor in BST
*/
class InorderSuccessor {
public:
    TreeNode* Solution(TreeNode* root, TreeNode* p)
    {
        // right exists, return the most left child of right
        if (p->right) return MostLeftChild(p->right);

        // no right, traverse the tree, successor is the first parent greater than p 
        // successor is the nearest parent if p in the left subtree
        // successor is NULL if p in the right subtree
        TreeNode* succ = NULL;
        while (root)
        {
            if (p->val < root->val)
            {
                succ = root;
                root = root->left;
            }
            else if (p->val > root->val) root = root->right;
            else break;
        }
        return succ;
    }
private:
    TreeNode* MostLeftChild(TreeNode* node)
    {
        while (node->left != NULL) node = node->left;
        return node;
    }
};

/*
297. Serialize and Deserialize Binary Tree
*/

class BinaryTreeCodec
{
    const char DELI = '|';
    const char END = '#';
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root)
    {
        stringstream nodes;

        PreorderEncode(root, nodes);
        return nodes.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data)
    {
        TreeNode *root = NULL;
        stringstream nodes(data);

        PreorderDecode(root, nodes);
        return root;
    }

private:
    void PreorderEncode(TreeNode* root, stringstream &nodes)
    {
        if (root == NULL)
        {
            nodes << END << DELI;
            return;
        }
        nodes << root->val << DELI;
        PreorderEncode(root->left, nodes);
        PreorderEncode(root->right, nodes);
    }

    void PreorderDecode(TreeNode* &curr, stringstream &nodes)
    {
        string str_val;
        if (!getline(nodes, str_val, DELI)) return;

        if (str_val[0] != END)
        {
            curr = new TreeNode(atoi(str_val.c_str()));
            PreorderDecode(curr->left, nodes);
            PreorderDecode(curr->right, nodes);
        }
    }
};

/*
472. Concatenated Words

Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

Input: ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
*/

// Memory Limit Exceeded!!!
class AllConcatenatedWordsInADict
{
    class Trie
    {
#define SET_SIZE 26
        struct Node
        {
            vector<Node*> CharSet;
            bool  IsWord;
            Node() :CharSet(SET_SIZE, nullptr), IsWord(false) {}
        };

    public:
        Trie() :Root_(new Node()) {}
        ~Trie() = default;

        void AddWord(const string &str)
        {
            Node *curr = Root_;
            for (char c : str)
            {
                if (curr->CharSet[c - 'a'] == nullptr)
                    curr->CharSet[c - 'a'] = new Node;
                curr = curr->CharSet[c - 'a'];
            }
            curr->IsWord = true;
        }

        bool CountWord(const string &str, int start, int count)
        {
            Node *curr = Root_;
            for (int i = start; i<str.size(); i++)
            {
                curr = curr->CharSet[str[i] - 'a'];
                if (curr == nullptr) return false;
                if (curr->IsWord)
                {
                    // last char
                    if (i == str.size() - 1) return count>0;
                    if (CountWord(str, i + 1, count + 1)) return true;
                }
            }
            return false;
        }

    private:
        Node *Root_;
    };

public:
    vector<string> Find(vector<string>& words)
    {
        vector<string> result;
        Trie pre_tree;
        for (auto &str : words)
            if (str.size()>0) pre_tree.AddWord(str);

        for (auto &str : words)
        {
            if (str.size() == 0) continue;

            if (pre_tree.CountWord(str, 0, 0)) result.push_back(str);
        }
        return result;
    }
};

#endif