Given an absolute path for a file (Unix-style), simplify it.

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"

class Solution {
public:
    string simplifyPath(string path) {
        vector<string> folders;
        int prev = -1;
        for(int index=0;index<path.size();index++){
            if(path[index] == '/' ){
                if(prev!=-1 && index-prev>1){
                    string sub = path.substr(prev+1,index-prev-1);
                    prev = index;
                    if(sub == "..") {
                        if(!folders.empty()) folders.pop_back();
                    }
                    else if(sub!=".") folders.push_back(sub);
                }
                prev = index;
            }
        }
        
        if(folders.size()==0) return "/";
        stringstream  ss;
        for(auto &name : folders)  ss<<"/"<<name;
        return ss.str();

    }
};
