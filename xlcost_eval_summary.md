# XLCoST Eval -- Hybrid Search

## Config
| Setting | Value |
|---|---|
| Collection | xlcost_repo |
| Top-K | 15 |
| Search Mode | hybrid |
| Queries per language | 100 |
| Total queries | 651 |

## Overall Metrics
| Metric | Score |
|---|---|
| Recall@15 | 0.962 |
| MRR | 0.719 |
| Hits | 626/651 |
| Errors | 0 |

## Per-Language Breakdown
| Language | Recall@15 | MRR | Hits |
|---|---|---|---|
| Python | 0.980 | 0.861 | 98/100 |
| Java | 0.980 | 0.766 | 98/100 |
| Cpp | 0.870 | 0.566 | 87/100 |
| Javascript | 0.960 | 0.630 | 96/100 |
| Csharp | 0.990 | 0.699 | 99/100 |
| PHP | 1.000 | 0.980 | 100/100 |
| C | 0.941 | 0.345 | 48/51 |

## Failures (first 30)
| Lang | Query | Target File | Top Retrieved |
|---|---|---|---|
| Python | Minimize count of increments of each element of subarrays re | Python/Python_0061.py | Python/Python_0033.py |
| Python | Number of coloured 0 's in an N | Function to return the cou | Python/Python_0093.py | PHP/PHP_0003.php |
| Java | Expected Number of Trials to get N Consecutive Heads | Java  | Java/Java_0064.java | Csharp/Csharp_0005.cs |
| Java | Count the numbers which can convert N to 1 using given opera | Java/Java_0071.java | Csharp/Csharp_0065.cs |
| Cpp | Insert minimum number in array so that sum of array becomes  | Cpp/Cpp_0007.cpp | Cpp/Cpp_0037.cpp |
| Cpp | Count ways to reach a score using 1 and 2 with no consecutiv | Cpp/Cpp_0008.cpp | PHP/PHP_0086.php |
| Cpp | Largest number with binary representation is m 1 's and m |  | Cpp/Cpp_0017.cpp | Python/Python_0003.py |
| Cpp | Palindromic strings of length 3 possible by using characters | Cpp/Cpp_0036.cpp | Python/Python_0057.py |
| Cpp | Minimize increments required to make count of even and odd a | Cpp/Cpp_0040.cpp | Python/Python_0033.py |
| Cpp | Nearest power of 2 of nearest perfect squares of non | C ++  | Cpp/Cpp_0046.cpp | Python/Python_0090.py |
| Cpp | Find minimum possible values of A , B and C when two of the  | Cpp/Cpp_0048.cpp | Python/Python_0054.py |
| Cpp | Check if an array is stack sortable | C ++ implementation of | Cpp/Cpp_0069.cpp | Java/Java_0047.java |
| Cpp | Count of all subsequence whose product is a Composite number | Cpp/Cpp_0071.cpp | Csharp/Csharp_0081.cs |
| Cpp | Minimum number of adjacent swaps required to convert a permu | Cpp/Cpp_0073.cpp | Csharp/Csharp_0027.cs |
| Cpp | Sort decreasing permutation of N using triple swaps | C ++ i | Cpp/Cpp_0082.cpp | PHP/PHP_0065.php |
| Cpp | Sum of even values and update queries on an array | C ++ imp | Cpp/Cpp_0086.cpp | Java/Java_0073.java |
| Cpp | Find a number such that maximum in array is minimum possible | Cpp/Cpp_0092.cpp | Java/Java_0093.java |
| Javascript | Check if a string has m consecutive 1 ' s ▁ or ▁ 0' s | Func | Javascript/Javascript_0004.js | PHP/PHP_0021.php |
| Javascript | Maximize the decimal equivalent by flipping only a contiguou | Javascript/Javascript_0031.js | PHP/PHP_0067.php |
| Javascript | Count Distinct Rectangles in N * N Chessboard | Function to  | Javascript/Javascript_0055.js | Javascript/Javascript_0033.js |
| Javascript | Minimum operations to make all elements equal using the seco | Javascript/Javascript_0076.js | Python/Python_0083.py |
| Csharp | Check if roots of a Quadratic Equation are reciprocal of eac | Csharp/Csharp_0020.cs | Csharp/Csharp_0006.cs |
| C | Count set bits in an integer |  ; Check each bit in a number | C/C_0002.c | Csharp/Csharp_0065.cs |
| C | Find all factors of a natural number | Set 1 | A Better ( th | C/C_0021.c | PHP/PHP_0046.php |
| C | Coin Change | DP | Recursive C program for coin change probl | C/C_0036.c | PHP/PHP_0076.php |
