#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<string> sequences(n);
    for (int i = 0; i < n; i++) {
        cin >> sequences[i];
    }
    
    // 对于每个序列，计算需要多少个左括号和右括号才能变成合法的
    vector<int> need_left(n, 0);  // 需要多少个左括号
    vector<int> need_right(n, 0); // 需要多少个右括号
    
    for (int i = 0; i < n; i++) {
        int left = 0, right = 0;
        for (char c : sequences[i]) {
            if (c == '(') {
                left++;
            } else { // c == ')'
                if (left > 0) {
                    left--; // 匹配一对
                } else {
                    right++; // 右括号无法匹配
                }
            }
        }
        need_left[i] = right;  // 需要right个左括号来匹配多余的右括号
        need_right[i] = left;  // 需要left个右括号来匹配多余的左括号
    }
    
    // 统计可以配对的序列
    int pairs = 0;
    
    // 情况1：需要左括号的序列和需要右括号的序列配对
    // 需要need_left[i] == need_right[j]且都大于0
    map<int, int> left_counts; // 统计需要左括号的序列数
    map<int, int> right_counts; // 统计需要右括号的序列数
    
    for (int i = 0; i < n; i++) {
        if (need_left[i] > 0 && need_right[i] == 0) {
            left_counts[need_left[i]]++;
        } else if (need_right[i] > 0 && need_left[i] == 0) {
            right_counts[need_right[i]]++;
        }
    }
    
    for (auto& p : left_counts) {
        int count = p.second;
        int right_count = right_counts[p.first];
        pairs += min(count, right_count);
    }
    
    // 情况2：两个序列都不需要额外括号（已经是合法的）
    int valid_count = 0;
    for (int i = 0; i < n; i++) {
        if (need_left[i] == 0 && need_right[i] == 0) {
            valid_count++;
        }
    }
    pairs += valid_count / 2;
    
    cout << pairs << endl;
    
    return 0;
}