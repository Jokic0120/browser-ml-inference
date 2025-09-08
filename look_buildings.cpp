#include <bits/stdc++.h>
using namespace std;

struct Building {
    int height;
    int color;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    if (!(cin >> t)) return 0;
    while (t--) {
        int n;
        cin >> n;
        vector<int> colors(n + 1);
        vector<int> heights(n + 1);
        for (int i = 1; i <= n; ++i) cin >> colors[i];
        for (int i = 1; i <= n; ++i) cin >> heights[i];

        // Colors are in [1, 1e6]
        const int MAX_COLOR = 1000000;
        vector<int> colorFrequency(MAX_COLOR + 1, 0);

        vector<int> result(n + 1, 0);
        vector<Building> stackMono;
        stackMono.reserve(n);

        int distinctColors = 0;
        for (int i = 1; i <= n; ++i) {
            int h = heights[i];
            int c = colors[i];

            // Pop buildings that are not visible anymore (height <= current)
            while (!stackMono.empty() && stackMono.back().height <= h) {
                int pc = stackMono.back().color;
                stackMono.pop_back();
                if (--colorFrequency[pc] == 0) {
                    --distinctColors;
                }
            }

            // Push current building
            stackMono.push_back({h, c});
            if (colorFrequency[c] == 0) ++distinctColors;
            ++colorFrequency[c];

            result[i] = distinctColors;
        }

        for (int i = 1; i <= n; ++i) {
            if (i > 1) cout << ' ';
            cout << result[i];
        }
        cout << '\n';
    }

    return 0;
}

