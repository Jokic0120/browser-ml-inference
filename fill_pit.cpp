#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    while (cin >> n) {
        vector<int> heights(n);
        for (int i = 0; i < n; i++) {
            cin >> heights[i];
        }
        
        // 找到最大高度
        int max_height = *max_element(heights.begin(), heights.end());
        
        // 计算每个位置需要填的高度
        vector<int> need(n);
        for (int i = 0; i < n; i++) {
            need[i] = max_height - heights[i];
        }
        
        // 检查是否能填平
        bool possible = true;
        
        // 从左到右贪心处理
        for (int i = 0; i < n; i++) {
            if (need[i] == 0) continue;
            
            // 如果当前位置需要奇数高度
            if (need[i] % 2 == 1) {
                if (i + 1 < n) {
                    // 用水平石头：当前位置+1，下一位置+1
                    need[i]--;
                    need[i + 1]--;
                } else {
                    // 没有下一位置，无法处理奇数
                    possible = false;
                    break;
                }
            }
            
            // 处理剩余的偶数高度
            while (need[i] > 0) {
                if (i + 1 < n && need[i + 1] > 0) {
                    // 优先用水平石头
                    int use = min(need[i], need[i + 1]);
                    need[i] -= use;
                    need[i + 1] -= use;
                } else {
                    // 只能用垂直石头
                    if (need[i] >= 2) {
                        need[i] -= 2;
                    } else {
                        possible = false;
                        break;
                    }
                }
            }
        }
        
        cout << (possible ? "YES" : "NO") << endl;
    }
    
    return 0;
}