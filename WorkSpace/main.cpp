#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <chrono>
using namespace std;
using namespace Eigen;
const int envx = 192;
const int envy = 108;
double armlength = envy / 4.0;
double armwidth = envy / 40.0;

typedef Matrix<bool, envx, envy> Space;
Space env = Space::Zero();
using Eigen::MatrixXd;

inline bool Inrange(int x)
{
    return x >= 0 && x < envx;
}

Space Presence(int x, int a1d, int a2d)
{
    // calculate second joint position
    Space pres = Space::Zero();
    int l = armlength;
    double a1 = a1d / 180 * 3.14;
    double a2 = a2d / 180 * 3.14;
    int w = armwidth;
    int x1 = x;
    int y1 = envy / 2;
    int x2 = x1 + l * cos(a1);
    int y2 = y1 + l * sin(a1);
    double d = 0;

    for (int x_ = 0; x_ < 2 * l; x_++)
    {
        for (int y_ = 0; y_ < 2 * l; y_++)
        {
            //arm 1
            int x = -l + x1 + x_;
            int y = -l + y1 + y_;
            if (Inrange(x) && y >= 0)
            {
                d = sqrt(pow(x - x1, 2) + pow(y - y1, 2));
                if (d < l)
                {
                    double A = tan(a1);
                    double dl = abs(A * x - y + y1 - x1 * A) / sqrt(pow(A, 2) + 1);
                    if (dl < w)
                    {
                        pres(x, y) = 1;
                        continue;
                    }
                }
            }
            //arm 2
            x = -l + x2 + x_;
            y = -l + y2 + y_;
            if (Inrange(x) && y >= 0)
            {
                d = sqrt(pow(x - x2, 2) + pow(y - y2, 2));
                if (d < l)
                {
                    double A = tan(a2);
                    double dl = abs(A * x - y + y2 - x2 * A) / sqrt(pow(A, 2) + 1);
                    if (dl < w)
                    {
                        pres(x, y) = 1;
                        continue;
                    }
                }
            }
        }
    }
    return pres;
}

bool Feasable(int x, int a1d, int a2d)
{
    Space overlap = Presence(x, a1d, a2d).cwiseProduct(env);
    return overlap.sum();
}

int main()
{
    chrono::steady_clock sc;
    auto start = sc.now();
    int dx = envx / 192;
    int da1 = 360 / 18;
    int da2 = 360 / 18;
    for (int nx = 0; nx < 192; nx++)
    {
        cout << nx << endl;
        for (int na1 = 0; na1 < 18; na1++)
        {
            for (int na2 = 0; na2 < 18; na2++)
            {
                int x = nx * dx;
                int a1d = -180 + na1 * da1;
                int a2d = -180 + na2 * da1;
                Feasable(x, a1d, a2d);
            }
        }
    }
    auto end = sc.now();
    auto time_span = static_cast<chrono::duration<double>>(end - start); // measure time span between start & end
    cout << "Operation took: " << time_span.count() << " seconds !!!";
    system("pause");
}