## Part 1
**Easy Level:**

1. **Problem:** Create a 1D NumPy array containing the first 10 positive integers.
   - **Input Data:** None
   - **Expected Output:** `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
   - **Hint:** Use `numpy.arange()`.

2. **Problem:** Create a 2x3 array with random values between 0 and 1.
   - **Input Data:** None
   - **Expected Output:** A 2x3 array with random values between 0 and 1.
   - **Hint:** Use `numpy.random.rand()`.

3. **Problem:** Calculate the sum of all elements in a 2D array.
   - **Input Data:**
     ```
     arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     ```
   - **Expected Output:** 45
   - **Hint:** Use `numpy.sum()`.

4. **Problem:** Create an identity matrix of size 4x4.
   - **Input Data:** None
   - **Expected Output:** A 4x4 identity matrix.
   - **Hint:** Use `numpy.eye()`.

5. **Problem:** Extract all even numbers from an array.
   - **Input Data:**
     ```
     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
     ```
   - **Expected Output:** `[2, 4, 6, 8]`
   - **Hint:** Use boolean indexing.

**Medium Level:**

6. **Problem:** Implement a function to compute the factorial of each element in a 1D array.
   - **Input Data:**
     ```
     arr = np.array([1, 2, 3, 4, 5])
     ```
   - **Expected Output:** Factorials of each element.
   - **Hint:** Use `numpy.math.factorial()`.

7. **Problem:** Create a function to calculate the moving average of a 1D array for a given window size.
   - **Input Data:**
     ```
     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
     window_size = 3
     ```
   - **Expected Output:** The moving average array.
   - **Hint:** Use `numpy.convolve()`.

8. **Problem:** Implement a function to find the median value of each row in a 2D array.
   - **Input Data:**
     ```
     arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     ```
   - **Expected Output:** Medians of each row.
   - **Hint:** Use `numpy.median()` with `axis=1`.

9. **Problem:** Create a function to compute the cross-correlation between two 1D arrays.
   - **Input Data:**
     ```
     x = np.array([1, 2, 1, 0, 1, 2, 1])
     y = np.array([1, -1, 0, 1])
     ```
   - **Expected Output:** The cross-correlation result.
   - **Hint:** Use `numpy.correlate()`.

10. **Problem:** Calculate the element-wise square root of a 2D array.
    - **Input Data:**
      ```
      arr = np.array([[1, 4, 9], [16, 25, 36]])
      ```
    - **Expected Output:** The square root of each element.
    - **Hint:** Use `numpy.sqrt()`.

**Hard Level:**

11. **Problem:** Implement a function to perform matrix multiplication without using `numpy.dot()` or the `@` operator.
    - **Input Data:**
      ```
      A = np.array([[1, 2], [3, 4]])
      B = np.array([[5, 6], [7, 8]])
      ```
    - **Expected Output:** The product of A and B.
    - **Hint:** Implement matrix multiplication manually using loops.

12. **Problem:** Create a function to find the k-nearest neighbors for each row in a 2D array, where k is a parameter.
    - **Input Data:**
      ```
      data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      k = 2
      ```
    - **Expected Output:** An array of indices for the k-nearest neighbors.
    - **Hint:** You can compute distances and use `numpy.argsort()`.

13. **Problem:** Calculate the Euclidean distance between two sets of 2D points, where each set is a 2D array.
    - **Input Data:**
      ```
      set1 = np.array([[1, 2], [3, 4]])
      set2 = np.array([[4, 3], [2, 1]])
      ```
    - **Expected Output:** An array of distances between points in set1 and set2.
    - **Hint:** Use `numpy.linalg.norm()`.

14. **Problem:** Implement a function to find the largest rectangle in a binary matrix (0s and 1s) using NumPy.
    - **Input Data:**
      ```
      matrix = np.array([[1, 0, 0, 1, 0],
                         [1, 0, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0]])
      ```
    - **Expected Output:** The area of the largest rectangle (in this case, 8).
    - **Hint:** Consider using dynamic programming.

15. **Problem:** Create a function to compute the 2D convolution of a 2D array and a 2D kernel.
    - **Input Data:**
      ```
      image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
      ```
    - **Expected Output:** The result of convolution.
    - **Hint:** Implement convolution using nested loops.

16. **Problem:** Implement a function to calculate the mean and standard deviation of each row in a 2D array.
    - **Input Data:**
      ```
      arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

**Easy Level:**

17. **Problem:** Create a 2D array with random integers between 1 and 100 (inclusive) and reshape it into a 5x5 matrix.
    - **Input Data:** None
    - **Expected Output:** A 5x5 matrix with random integers.
    - **Hint:** Use `numpy.random.randint()` and `numpy.reshape()`.

18. **Problem:** Create a function to find the unique elements and their counts in a 1D array.
    - **Input Data:**
      ```
      arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
      ```
    - **Expected Output:** Unique elements and their counts.
    - **Hint:** Use `numpy.unique()` with `return_counts=True`.

19. **Problem:** Create a function to calculate the element-wise minimum and maximum of two arrays.
    - **Input Data:**
      ```
      A = np.array([1, 2, 3, 4])
      B = np.array([3, 2, 1, 5])
      ```
    - **Expected Output:** Element-wise minimum and maximum arrays.
    - **Hint:** Use `numpy.minimum()` and `numpy.maximum()`.

20. **Problem:** Create a 1D array containing the first 20 odd positive integers.
    - **Input Data:** None
    - **Expected Output:** `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]`
    - **Hint:** Use `numpy.arange()` with a step size of 2.

**Medium Level:**

21. **Problem:** Implement a function to calculate the exponential moving average of a 1D array with a specified smoothing factor.
    - **Input Data:**
      ```
      arr = np.array([10, 12, 15, 20, 18, 22, 25, 30, 28, 35])
      smoothing_factor = 0.2
      ```
    - **Expected Output:** The exponential moving average array.
    - **Hint:** Use a recursive formula to update the moving average.

22. **Problem:** Create a function to calculate the eigenvalues and eigenvectors of a 3x3 matrix.
    - **Input Data:**
      ```
      A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      ```
    - **Expected Output:** The eigenvalues and eigenvectors.
    - **Hint:** Use `numpy.linalg.eig()`.

23. **Problem:** Implement a function to compute the correlation matrix of a 2D dataset.
    - **Input Data:**
      ```
      data = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
      ```
    - **Expected Output:** The correlation matrix.
    - **Hint:** Use `numpy.corrcoef()`.

24. **Problem:** Calculate the determinant of a 4x4 matrix without using `numpy.linalg.det()`.
    - **Input Data:**
      ```
      A = np.array([[2, 1, 4, 3], [4, 2, 5, 7], [3, 1, 2, 8], [9, 3, 1, 6]])
      ```
    - **Expected Output:** The determinant of matrix A.
    - **Hint:** Implement the determinant calculation manually.

25. **Problem:** Implement a function to find the index of the second largest element in a 1D array.
    - **Input Data:**
      ```
      arr = np.array([7, 2, 9, 4, 8, 6])
      ```
    - **Expected Output:** The index of the second largest element (0-based index).
    - **Hint:** Use `numpy.argsort()` and select the second-to-last index.

**Hard Level:**

26. **Problem:** Implement a function to calculate the inverse of a 3x3 matrix without using `numpy.linalg.inv()`.
    - **Input Data:**
      ```
      A = np.array([[2, 1, 3], [1, 3, 4], [5, 6, 2]])
      ```
    - **Expected Output:** The inverse of matrix A.
    - **Hint:** Implement the formula for the inverse of a 3x3 matrix.

27. **Problem:** Create a function to find the largest prime number in a 2D array of integers.
    - **Input Data:**
      ```
      matrix = np.array([[7, 12, 15], [4, 19, 6], [10, 8, 5]])
      ```
    - **Expected Output:** The largest prime number in the matrix.
    - **Hint:** Implement a function to check for prime numbers.

28. **Problem:** Implement a function to solve a system of linear equations represented as matrices Ax = b.
    - **Input Data:**
      ```
      A = np.array([[2, 1, -1], [1, 3, 2], [3, 2, 0]])
      b = np.array([8, 9, 3])
      ```
    - **Expected Output:** The solution vector x.
    - **Hint:** Use `numpy.linalg.solve()`.

29. **Problem:** Create a function to compute the discrete Fourier transform (DFT) of a 1D array.
    - **Input Data:**
      ```
      arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
      ```
    - **Expected Output:** The DFT of the input array.
    - **Hint:** Implement the DFT formula manually.

30. **Problem:** Implement a function to calculate the distance between two points on the Earth's surface given their longitude and latitude coordinates.
    - **Input Data:**
      ```
      coordinates1 = np.array([40.7128, -74.0060])  # New York
      coordinates2 = np.array([34.0522, -118.2437])  # Los Angeles
      ```
    - **Expected Output:** The distance in kilometers between the two locations.
    - **Hint:** Use the Haversine formula for spherical distance.

**Easy Level:**

31. **Problem:** Create a 1D array of 20 equally spaced values from 0 to 1.
   - **Input Data:** None
   - **Expected Output:** An array of 20 values.
   - **Hint:** Use `numpy.linspace()`.

32. **Problem:** Create a function that computes the mean and median of a 1D array and returns the results.
   - **Input Data:**
     ```
     arr = np.array([12, 34, 45, 23, 67, 89, 33, 21, 56, 78])
     ```
   - **Expected Output:** Mean and median of the array.
   - **Hint:** Use `numpy.mean()` and `numpy.median()`.

33. **Problem:** Implement a function to calculate the sum of the diagonal elements in a square matrix.
   - **Input Data:**
     ```
     matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     ```
   - **Expected Output:** The sum of the diagonal elements (15 in this case).
   - **Hint:** Use indexing to access diagonal elements.

34. **Problem:** Create a function to count the number of elements greater than a given threshold in a 1D array.
   - **Input Data:**
     ```
     arr = np.array([12, 34, 45, 23, 67, 89, 33, 21, 56, 78])
     threshold = 50
     ```
   - **Expected Output:** The count of elements greater than the threshold.
   - **Hint:** Use boolean indexing.

35. **Problem:** Find the index of the minimum element in a 1D array.
   - **Input Data:**
     ```
     arr = np.array([7, 2, 9, 4, 8, 6])
     ```
   - **Expected Output:** The index of the minimum element (1 in this case).
   - **Hint:** Use `numpy.argmin()`.

**Medium Level:**

36. **Problem:** Create a function to calculate the pairwise Euclidean distances between points in two arrays, A and B.
   - **Input Data:**
     ```
     A = np.array([[1, 2], [3, 4]])
     B = np.array([[4, 3], [2, 1]])
     ```
   - **Expected Output:** A matrix of distances between points in A and B.
   - **Hint:** Use nested loops and `numpy.linalg.norm()`.

37. **Problem:** Implement a function to find the index of the smallest element greater than a given threshold in a 1D array.
   - **Input Data:**
     ```
     arr = np.array([12, 34, 45, 23, 67, 89, 33, 21, 56, 78])
     threshold = 50
     ```
   - **Expected Output:** The index of the smallest element greater than the threshold (9 in this case).
   - **Hint:** Use boolean indexing and `numpy.argmin()`.

38. **Problem:** Create a function to calculate the cumulative sum of a 1D array, but with a twist: the sum resets whenever a negative value is encountered.
   - **Input Data:**
     ```
     arr = np.array([1, 2, -1, 3, 4, -2, 5, 6])
     ```
   - **Expected Output:** The cumulative sum with resets: `[1, 3, 0, 3, 7, 0, 5, 11]`
   - **Hint:** Use a loop to keep track of the cumulative sum and reset when needed.

39. **Problem:** Create a function to find the majority element in a 1D array, which appears more than n/2 times (n is the length of the array).
   - **Input Data:**
     ```
     arr = np.array([2, 2, 1, 1, 1, 2, 2])
     ```
   - **Expected Output:** The majority element (2 in this case).
   - **Hint:** Implement a counter and check for majority.

40. **Problem:** Implement a function to calculate the product of all non-zero elements in a 1D array.
    - **Input Data:**
      ```
      arr = np.array([1, 2, 0, 4, 0, 3, 5, 0])
      ```
    - **Expected Output:** The product of non-zero elements (120 in this case).
    - **Hint:** Use boolean indexing and `numpy.prod()`.

**Hard Level:**

41. **Problem:** Create a function to perform matrix multiplication for two non-square matrices (A and B) using NumPy.
   - **Input Data:**
     ```
     A = np.array([[1, 2, 3], [4, 5, 6]])
     B = np.array([[7, 8], [9, 10], [11, 12]])
     ```
   - **Expected Output:** The product of matrices A and B.
   - **Hint:** Implement matrix multiplication using loops.

42. **Problem:** Implement a function to find the k-means clustering of a set of data points in a 2D array.
   - **Input Data:**
     ```
     data = np.array([[2, 3], [3, 4], [3, 5], [5, 5], [6, 6], [7, 8]])
     k = 2
     ```
   - **Expected Output:** Cluster assignments for each data point.
   - **Hint:** Implement the k-means algorithm with random initial centroids.

43. **Problem:** Create a function to compute the product of the diagonal elements in a square matrix.
   - **Input Data:**
     ```
     matrix = np.array([[2, 1, 4], [3, 2, 5], [6, 7, 3]])
     ```
   - **Expected Output:** The product of diagonal elements (18 in this case).
   - **Hint:** Use indexing to access diagonal elements.

44. **Problem:** Implement a function to compute the cosine similarity between two sets of vectors (A and B).
   - **Input Data:**
     ```
     A = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
     B = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
     ```
   - **Expected Output:** A cosine similarity matrix.
   - **Hint:** Normalize vectors and use `numpy.dot()`.

45. **Problem:** Create a function to compute the n-th Fibonacci number using matrix exponentiation.
    - **Input Data:**
      ```
      n = 10
      ```
    - **Expected Output:** The n-th Fibonacci number.
    - **Hint:** Use matrix exponentiation and the properties of the Fibonacci sequence.

## Part 2

1. Problem: Create an array of zeros with 5 rows and 3 columns.
   Input: None
   Expected Output: 
   ```
   array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]])
   ```

2. Problem: Create an array with values ranging from 10 to 49.
   Input: None
   Expected Output: 
   ```
   array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
   ```

3. Problem: Reverse the order of elements in an array.
   Input: `arr = np.array([1, 2, 3, 4, 5])`
   Expected Output: 
   ```
   array([5, 4, 3, 2, 1])
   ```

4. Problem: Create a 3x3 matrix with values ranging from 0 to 8.
   Input: None
   Expected Output: 
   ```
   array([[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]])
   ```

5. Problem: Find the indices of non-zero elements in an array.
   Input: `arr = np.array([0, 2, 0, 4, 0, 6])`
   Expected Output: `(array([1, 3, 5]),)`

6. Problem: Create a 3x3 identity matrix.
   Input: None
   Expected Output: 
   ```
   array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.]])
   ```

7. Problem: Create a 3x3x3 array with random values.
   Input: None
   Expected Output: (Output values will vary)

8. Problem: Create a 5x5 matrix with values 1, 2, 3, 4 just below the diagonal.
   Input: None
   Expected Output: 
   ```
   array([[0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0.],
          [0., 2., 0., 0., 0.],
          [0., 0., 3., 0., 0.],
          [0., 0., 0., 4., 0.]])
   ```

9. Problem: Create a 8x8 matrix and fill it with a checkerboard pattern (0s and 1s).
   Input: None
   Expected Output: 
   ```
   array([[0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
          [1, 0, 1, 0, 1, 0, 1, 0]])
   ```

10. Problem: Create a random 5x5 array and normalize it (scale it so that the values range from 0 to 1).
    Input: None
    Expected Output: (Output values will vary)

Hints:
- For problems 1-6, you can use `np.zeros`, `np.arange`, `np.reshape`, `np.identity`, and `np.random.rand`.
- For problem 3, use array slicing.
- For problem 5, use `np.nonzero`.
- For problems 8-9, use slicing and assignment.

11. Problem: Create a 2D array with 1s on the border and 0s inside.
    Input: None
    Expected Output:
    ```
    array([[1., 1., 1., 1., 1.],
           [1., 0., 0., 0., 1.],
           [1., 0., 0., 0., 1.],
           [1., 0., 0., 0., 1.],
           [1., 1., 1., 1., 1.]])
    ```

12. Problem: Create a 2D array with random values and find the minimum and maximum values.
    Input: None
    Expected Output: (Output values will vary)

13. Problem: Create a random vector of size 30 and find the mean value.
    Input: None
    Expected Output: (Output value will vary)

14. Problem: Create a 5x5 matrix with values 1,2,3,4 just above the diagonal.
    Input: None
    Expected Output:
    ```
    array([[0, 1, 2, 0, 0],
           [0, 0, 1, 2, 0],
           [0, 0, 0, 1, 2],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]])
    ```

15. Problem: Create a 2D array of shape (5, 5) with 1's on the diagonal and 0's elsewhere.
    Input: None
    Expected Output:
    ```
    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])
    ```

16. Problem: Create a 5x5 matrix with values 1,2,3,4 just below the diagonal.
    Input: None
    Expected Output:
    ```
    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 4, 0]])
    ```

17. Problem: Given a 1D array, negate all elements which are between 3 and 8 (inclusive).
    Input: `arr = np.array([1, 5, 4, 9, 7, 2, 6])`
    Expected Output:
    ```
    array([ 1, -5, -4,  9, -7,  2, -6])
    ```

18. Problem: Create a 5x5 matrix with row values ranging from 0 to 4.
    Input: None
    Expected Output:
    ```
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    ```

19. Problem: Multiply a 5x3 matrix by a 3x2 matrix (matrix multiplication).
    Input: None (create random matrices)
    Expected Output: (Output values will vary)

20. Problem: Calculate the dot product of two arrays.
    Input: `arr1 = np.array([1, 2, 3])`, `arr2 = np.array([4, 5, 6])`
    Expected Output: `32`

Hints:
- For problem 11, you can create a matrix of ones and then set the interior values to zero.
- For problem 12, use `np.min` and `np.max`.
- For problem 14 and 16, use slicing and assignment.
- For problem 17, use boolean indexing.
- For problem 18, you can use broadcasting.
- For problems 19 and 20, use `np.dot` or the `@` operator for matrix multiplication.

21. Problem: Create a random 3x3 matrix and sort the columns in ascending order.
    Input: None
    Expected Output: (Output values will vary)

22. Problem: Given a 1D array, replace all the odd numbers with -1.
    Input: `arr = np.array([1, 2, 3, 4, 5, 6, 7])`
    Expected Output: `array([-1,  2, -1,  4, -1,  6, -1])`

23. Problem: Reshape a 1D array into a 2D array with 2 rows.
    Input: `arr = np.array([1, 2, 3, 4, 5, 6])`
    Expected Output:
    ```
    array([[1, 2, 3],
           [4, 5, 6]])
    ```

24. Problem: Given two arrays, concatenate them along the second axis.
    Input: `arr1 = np.array([1, 2, 3])`, `arr2 = np.array([4, 5, 6])`
    Expected Output:
    ```
    array([[1, 4],
           [2, 5],
           [3, 6]])
    ```

25. Problem: Create a random 5x5 matrix and normalize it row-wise.
    Input: None
    Expected Output: (Output values will vary)

26. Problem: Create a 1D array of random integers between 1 and 100, and filter out values greater than 50.
    Input: None
    Expected Output: (Output values will vary)

27. Problem: Create a random 4x4 matrix and extract its diagonal as a 1D array.
    Input: None
    Expected Output: (Output values will vary)

28. Problem: Find the unique values and their counts in a 1D array.
    Input: `arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])`
    Expected Output:
    ```
    (array([1, 2, 3, 4]), array([1, 2, 3, 4]))
    ```

29. Problem: Create a 2D array with 1s on the border and 0s inside (excluding the border).
    Input: None
    Expected Output:
    ```
    array([[1., 1., 1., 1.],
           [1., 0., 0., 1.],
           [1., 0., 0., 1.],
           [1., 1., 1., 1.]])
    ```

30. Problem: Given a 1D array, remove all elements which are equal to the mean of the array.
    Input: `arr = np.array([1, 2, 3, 4, 5, 6, 7])`
    Expected Output: `array([1, 2, 3, 5, 6, 7])`

Hints:
- For problem 21, use `np.sort` with the appropriate axis argument.
- For problem 23, use `np.reshape`.
- For problem 24, use `np.concatenate` with `axis=1`.
- For problem 25, use broadcasting.
- For problem 28, use `np.unique`.
- For problem 29, set the interior values to zero using slicing.

31. Problem: Create a 5x5 matrix with random values and find the sum of its diagonal elements.
    Input: None
    Expected Output: (Output value will vary)

32. Problem: Create a random 4x4 matrix and replace the maximum value by -1 and the minimum value by 1.
    Input: None
    Expected Output: (Output values will vary)

33. Problem: Given a 1D array, find the number of elements between a given range, say, 5 and 10 (inclusive).
    Input: `arr = np.array([1, 6, 8, 12, 5, 15, 9, 7])`
    Expected Output: `4`

34. Problem: Create a 5x5 matrix with random values and normalize it along its rows (row-wise normalization).
    Input: None
    Expected Output: (Output values will vary)

35. Problem: Given two 1D arrays, find their common elements.
    Input: `arr1 = np.array([1, 2, 3, 4, 5])`, `arr2 = np.array([4, 5, 6, 7, 8])`
    Expected Output: `array([4, 5])`

36. Problem: Create a 3x3 matrix with random values and find the row with the maximum sum.
    Input: None
    Expected Output: (Output values will vary)

37. Problem: Create a random 3x3 matrix and negate all elements between the 25th and 75th percentile (inclusive).
    Input: None
    Expected Output: (Output values will vary)

38. Problem: Calculate the mean squared error (MSE) between two 1D arrays, `arr1` and `arr2`.
    Input: `arr1 = np.array([1, 2, 3, 4, 5])`, `arr2 = np.array([2, 2, 3, 3, 5])`
    Expected Output: `0.6`

39. Problem: Create a 1D array of random integers between 1 and 100, and find the longest sequence of consecutive numbers.
    Input: None
    Expected Output: (Output values will vary)

40. Problem: Given a 2D array, find the index of the row with the most 1s.
    Input: `arr = np.array([[0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 0], [1, 1, 0, 0, 1]])`
    Expected Output: `1`

