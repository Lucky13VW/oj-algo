class LRUCache {
    struct Node{
        int key;
        int value;
        Node(int k,int v):
        key(k),value(v){}
    };
    
public:
    LRUCache(int capacity)
        :Capacity_(capacity),
    {}
    
    int get(int key) {
        auto it = Index_.find(key);
        if(it==Index_.end()) return -1;
        // move the hit node to begin position
        Cache_.splice(Cache_.begin(),Cache_,it->second);
        return it->second->value;
    }
    
    void put(int key, int value) {
        auto map_it = Index_.find(key);
        // only update value
        if(map_it != Index_.end()) map_it->second->value = value;
        else{
            // remove the LRU node from both list and hashtable
            if(Index_.size()==Capacity_)
            {
                int k = Cache_.back()->key;
                Cache_.pop_back();
                Index_.erase(Index_.find(k));
            }
            Cache_.push_back(Node(key,value));
            auto list_it = Cache_.back();
            Index_.insert(pair<int,List<Node>>(key,list_it));
        }        
    }
    
private:
    int Capacity_;
    unordered_map<int,List<Node>::iterator> Index_;
    list<Node> Cache_;
};
