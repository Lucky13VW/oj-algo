#ifndef LEETCODE_LIST_HPP
#define LEETCODE_LIST_HPP

#include "leetcode.hpp"

using namespace std;

/**************************************************
                   List
***************************************************/

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

/*
21. Merge Two Sorted Lists
Merge two sorted linked lists and return it as a new list.
The new list should be made by splicing together the nodes of the first two lists.
*/
class MergeTwoSortedLists
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
23. Merge k Sorted Lists
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
*/
class MergekSortedLists
{
    class MinHeap
    {
    public:
        MinHeap(size_t cap) :_Capacity(cap + 1)
        {
            _Data.push_back(NULL); // dummy index 0
        }

        void Push(ListNode *node)
        {
            if (_Data.size()<_Capacity) _Data.push_back(node);
            else _Data[_Data.size() - 1] = node;

            Swim(_Data.size() - 1);
        }

        void Pop()
        {
            _Data[1] = _Data[_Data.size() - 1];
            _Data.pop_back();
            Sink(1);
        }

        ListNode *Top()
        {
            return _Data.size()>1 ? _Data[1] : NULL;
        }

        bool Empty() const { return _Data.size() == 1; }

    private:
        void Swim(int index)
        {
            if (index == 0) return;

            int parent = index / 2;
            if (parent == 0) return;

            if (_Data[index]->val >= _Data[parent]->val) return;

            swap(_Data[index], _Data[parent]);
            Swim(parent);
        }

        void Sink(int index)
        {
            int left = index * 2;
            if (left > _Data.size() - 1) return;

            int child = left;
            if (left + 1 < _Data.size() && _Data[left + 1]->val<_Data[left]->val) child = left + 1;

            if (_Data[child]->val >= _Data[index]->val) return;

            swap(_Data[child], _Data[index]);
            Sink(child);
        }

    private:
        size_t _Capacity;
        vector<ListNode*> _Data;
    };
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.size() == 0) return NULL;

        ListNode *head = NULL, *prev = NULL;
        MinHeap min_heap(lists.size());
        for (auto node : lists)
            if (node != NULL) min_heap.Push(node);

        while (!min_heap.Empty()) {
            ListNode *node = min_heap.Top();
            min_heap.Pop();
            if (prev == NULL) head = prev = node;
            else {
                prev->next = node;
                prev = node;
            }
            if (node->next != NULL) min_heap.Push(node->next);
        }

        return head;
    }
};


/*
24. Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head.
e.g:
Given 1->2->3->4, you should return the list as 2->1->4->3.
You may not modify the values in the list, only nodes itself can be changed.
*/
class SwapNodesInPairs {
public:
    ListNode* swapPairs(ListNode* head)
    {
        if (head == NULL) return NULL;

        ListNode *prev = NULL, *curr = head, *next = curr->next;
        while (next != NULL)
        {
            ListNode *temp = next->next;
            next->next = curr;
            curr->next = temp;
            if (prev != NULL) prev->next = next;
            else head = next;

            // next already swapped with curr
            prev = curr;
            curr = curr->next;
            if (curr == NULL) break;
            next = curr->next;
        }

        return head;
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
160. Intersection of Two Linked Lists
*/
class IntersectionTwoLinkedLists
{
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int lenA = 0, lenB = 0;
        ListNode *currA = headA, *currB = headB;
        while (currA != NULL) { lenA++; currA = currA->next; }
        while (currB != NULL) { lenB++; currB = currB->next; }

        if (lenA>lenB)
            for (int i = 0; i<lenA - lenB; i++) headA = headA->next;
        if (lenB>lenA)
            for (int i = 0; i<lenB - lenA; i++) headB = headB->next;

        while (headA != NULL && headB != NULL)
        {
            if (headA == headB) return headA;
            else { headA = headA->next; headB = headB->next; }
        }
        return NULL;
    }

    // If one of them reaches the end earlier then reuse it 
    // by moving it to the beginning of other list.
    // Once both of them go through reassigning, 
    // they will be equidistant from the collision point.
    ListNode *getIntersectionNodeV2(ListNode *headA, ListNode *headB)
    {
        if (headA == NULL || headB == NULL) return NULL;

        ListNode *pA = headA, *pB = headB;
        while (pA != pB)
        {
            pA = pA == NULL ? headB : pA->next;
            pB = pB == NULL ? headA : pB->next;
        }
        return pA;
    }
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



#endif