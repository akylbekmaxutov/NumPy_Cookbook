## Easy

1. **Problem:** Create an array of integers from 1 to 10.
   - **Data:** None
   - **Expected Output:** `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
   - **Hint:** Use `numpy.arange()`.

2. **Problem:** Create a 3x3 matrix with values ranging from 0 to 8.
   - **Data:** None
   - **Expected Output:**
     ```
     [[0, 1, 2],
      [3, 4, 5],
      [6, 7, 8]]
     ```
   - **Hint:** Use `numpy.arange()` and `numpy.reshape()`.

3. **Problem:** Create a 5x5 identity matrix.
   - **Data:** None
   - **Expected Output:**
     ```
     [[1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1]]
     ```
   - **Hint:** Use `numpy.eye()`.

4. **Problem:** Multiply a 3x3 matrix by a 3x3 identity matrix.
   - **Data:**
     ```
     A = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
     I = # Identity matrix
     ```
   - **Expected Output:**
     ```
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]
     ```
   - **Hint:** Use matrix multiplication.

5. **Problem:** Extract all odd numbers from an array.
   - **Data:**
     ```
     arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
     ```
   - **Expected Output:** `[1, 3, 5, 7, 9]`
   - **Hint:** Use array slicing and boolean indexing.

6. **Problem:** Replace all even numbers in an array with -1.
   - **Data:**
     ```
     arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
     ```
   - **Expected Output:** `[1, -1, 3, -1, 5, -1, 7, -1, 9]`
   - **Hint:** Use boolean indexing.

7. **Problem:** Reshape a 1D array into a 2D array (matrix).
   - **Data:**
     ```
     arr = [1, 2, 3, 4, 5, 6]
     ```
   - **Expected Output:**
     ```
     [[1, 2],
      [3, 4],
      [5, 6]]
     ```
   - **Hint:** Use `numpy.reshape()`.

8. **Problem:** Compute the mean and median of an array.
   - **Data:**
     ```
     arr = [5, 2, 7, 1, 8, 4]
     ```
   - **Expected Output:**
     - Mean: 4.5
     - Median: 4.5
   - **Hint:** Use `numpy.mean()` and `numpy.median()`.

9. **Problem:** Find the maximum and minimum values in an array.
   - **Data:**
     ```
     arr = [12, 3, 45, 6, 78, 90, 23]
     ```
   - **Expected Output:**
     - Maximum: 90
     - Minimum: 3
   - **Hint:** Use `numpy.max()` and `numpy.min()`.

10. **Problem:** Calculate the dot product of two arrays.
    - **Data:**
      ```
      A = [1, 2, 3]
      B = [4, 5, 6]
      ```
    - **Expected Output:** 32 (1*4 + 2*5 + 3*6)
    - **Hint:** Use `numpy.dot()` or simply use the `@` operator.

11. **Problem:** Create an array of zeros with shape (3, 4).
      - **Input Data:** None
      - **Expected Output:**
        ```
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        ```
      - **Hint:** Use `numpy.zeros()`.

12. **Problem:** Create an array of ones with shape (2, 3, 4).
      - **Input Data:** None
      - **Expected Output:**
        ```
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
        ```
      - **Hint:** Use `numpy.ones()`.

13. **Problem:** Create an array of evenly spaced values from 1 to 10 (inclusive) with a step size of 0.5.
      - **Input Data:** None
      - **Expected Output:**
        ```
        array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5, 10.])
        ```
      - **Hint:** Use `numpy.arange()` with a step size.

14. **Problem:** Create a 3x3 diagonal matrix with the values 1, 2, and 3 on the main diagonal.
      - **Input Data:** None
      - **Expected Output:**
        ```
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
        ```
      - **Hint:** Use `numpy.diag()`.

15. **Problem:** Stack two arrays vertically.
      - **Input Data:**
        ```
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        ```
      - **Expected Output:**
        ```
        array([[1, 2, 3],
               [4, 5, 6]])
        ```
      - **Hint:** Use `numpy.vstack()` or `numpy.concatenate()` with `axis=0`.

16. **Problem:** Calculate the sum of all elements in a 2D array.
      - **Input Data:**
        ```
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ```
      - **Expected Output:** 45
      - **Hint:** Use `numpy.sum()`.

17. **Problem:** Normalize the values in a 1D array to have a mean of 0 and a standard deviation of 1.
      - **Input Data:**
        ```
        arr = np.array([10, 20, 30, 40, 50])
        ```
      - **Expected Output:** The normalized array.
      - **Hint:** Subtract the mean and divide by the standard deviation.

18. **Problem:** Find the unique elements and their counts in a 1D array.
      - **Input Data:**
        ```
        arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        ```
      - **Expected Output:** Unique elements and their counts.
      - **Hint:** Use `numpy.unique()` with `return_counts=True`.

19. **Problem:** Calculate the element-wise square root of a 1D array.
      - **Input Data:**
        ```
        arr = np.array([1, 4, 9, 16, 25])
        ```
      - **Expected Output:** The square root of each element.
      - **Hint:** Use `numpy.sqrt()`.

20. **Problem:** Find the index of the maximum element in a 1D array.
       - **Input Data:**
         ```
         arr = np.array([7, 2, 9, 4, 8, 6])
         ```
       - **Expected Output:** The index of the maximum element (2 in this case).
       - **Hint:** Use `numpy.argmax()`.

## Medium

1. **Problem:** Implement a function to find the local maxima in a 1D array (peaks).
   - **Input Data:**
     ```
     arr = np.array([1, 3, 7, 1, 2, 6, 4, 8, 1])
     ```
   - **Expected Output:** An array of peak indices: `[2, 5, 7]`
   - **Hint:** Look for values where the neighboring elements are smaller.

2. **Problem:** Calculate the moving average of a 1D array with a specified window size, but this time use a sliding window with equal weights.
   - **Input Data:**
     ```
     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
     window_size = 3
     ```
   - **Expected Output:** The moving average array with the sliding window.
   - **Hint:** Use `numpy.convolve()` with a window of `[1/3, 1/3, 1/3]`.

3. **Problem:** Create a function to compute the cross-correlation between two 1D arrays.
   - **Input Data:**
     ```
     x = np.array([1, 2, 1, 0, 1, 2, 1])
     y = np.array([1, -1, 0, 1])
     ```
   - **Expected Output:** The cross-correlation result.
   - **Hint:** Use `numpy.correlate()`.

4. **Problem:** Implement a function to find the largest rectangle in a binary matrix (0s and 1s) using NumPy.
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

5. **Problem:** Implement a function to compute the 2D convolution of a 2D array and a 2D kernel.
   - **Input Data:**
     ```
     image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
     ```
   - **Expected Output:** The result of convolution.
   - **Hint:** Implement convolution using nested loops.

6. **Problem:** Create a function to find the percentile values (e.g., 25th and 75th percentiles) of a 1D array.
   - **Input Data:**
     ```
     arr = np.array([12, 34, 45, 23, 67, 89, 33, 21, 56, 78])
     ```
   - **Expected Output:** The 25th and 75th percentiles.
   - **Hint:** Use `numpy.percentile()`.

7. **Problem:** Implement a function to find the eigenvalues and eigenvectors of a 4x4 matrix.
   - **Input Data:**
     ```
     A = np.array([[1, 2, 3, 4],
                  [4, 5, 6, 7],
                  [7, 8, 9, 10],
                  [10, 11, 12, 13]])
     ```
   - **Expected Output:** The eigenvalues and eigenvectors.
   - **Hint:** Use `numpy.linalg.eig()`.

8. **Problem:** Create a function to generate a random 3x3 matrix and compute its singular value decomposition (SVD).
   - **Input Data:** None (random matrix generation).
   - **Expected Output:** The original matrix, and matrices U, S, and V in the SVD.
   - **Hint:** Use `numpy.linalg.svd()`.

9. **Problem:** Implement a function to calculate the mean and standard deviation of each row in a 2D array.
   - **Input Data:**
     ```
     arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
     ```
   - **Expected Output:** Mean and standard deviation for each row.
   - **Hint:** Use `numpy.mean()` and `numpy.std()` with `axis=1`.

10. **Problem:** Create a function to compute the element-wise minimum and maximum of two arrays.
    - **Input Data:**
      ```
      A = np.array([1, 2, 3, 4])
      B = np.array([3, 2, 1, 5])
      ```
    - **Expected Output:** Element-wise minimum and maximum arrays.
    - **Hint:** Use `numpy.minimum()` and `numpy.maximum()`.

## Hard

1. **Problem:** Find the Euclidean distance between two arrays, A and B, of shape (n, m), where n is the number of points, and m is the number of dimensions.
   - **Input Data:**
     ```
     A = np.array([[1, 2], [3, 4]])
     B = np.array([[4, 3], [2, 1]])
     ```
   - **Expected Output:** An array of distances: `[2.82842712, 2.82842712]`
   - **Hint:** Use `numpy.linalg.norm()`.

2. **Problem:** Given a 2D array, find the k-nearest neighbors for each row, where k is a parameter.
   - **Input Data:**
     ```
     data = np.array([[1, 2], [4, 5], [7, 8], [10, 11]])
     k = 2
     ```
   - **Expected Output:** An array of indices for the k-nearest neighbors.
   - **Hint:** You can compute distances and use `numpy.argsort()`.

3. **Problem:** Calculate the cosine similarity between all pairs of rows in a 2D array.
   - **Input Data:**
     ```
     data = np.array([[1, 2], [3, 4], [5, 6]])
     ```
   - **Expected Output:** A cosine similarity matrix.
   - **Hint:** Normalize rows and use `numpy.dot()`.

4. **Problem:** Implement a function that performs matrix multiplication without using `numpy.dot()` or the `@` operator.
   - **Input Data:**
     ```
     A = np.array([[1, 2], [3, 4]])
     B = np.array([[5, 6], [7, 8]])
     ```
   - **Expected Output:** The product of A and B.
   - **Hint:** Implement matrix multiplication manually using loops.

5. **Problem:** Reshape an array so that its shape is the reverse of the original shape.
   - **Input Data:**
     ```
     arr = np.array([[1, 2], [3, 4], [5, 6]])
     ```
   - **Expected Output:** A reshaped array with shape (2, 3).
   - **Hint:** Use `numpy.reshape()` and `-1` for automatic dimension calculation.

6. **Problem:** Create a function that computes the moving average of a 1D array for a given window size.
   - **Input Data:**
     ```
     arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
     window_size = 3
     ```
   - **Expected Output:** The moving average array.
   - **Hint:** Use `numpy.convolve()`.

7. **Problem:** Implement a function to find the closest pair of points in a set of 2D points using NumPy.
   - **Input Data:**
     ```
     points = np.array([[1, 2], [4, 5], [7, 8], [10, 11]])
     ```
   - **Expected Output:** The closest pair of points and their distance.
   - **Hint:** Use a distance matrix and `numpy.argmin()`.

8. **Problem:** Create a function to calculate the inverse of a 2x2 matrix without using `numpy.linalg.inv()`.
   - **Input Data:**
     ```
     A = np.array([[2, 1], [1, 3]])
     ```
   - **Expected Output:** The inverse of the matrix A.
   - **Hint:** Implement the formula for the inverse of a 2x2 matrix.

9. **Problem:** Perform element-wise matrix exponentiation for a given 2x2 matrix and an exponent.
   - **Input Data:**
     ```
     A = np.array([[2, 3], [1, 4]])
     exponent = 3
     ```
   - **Expected Output:** The result of A raised to the power of 3.
   - **Hint:** Implement the exponentiation using matrix multiplication.

10. **Problem:** Calculate the eigenvalues and eigenvectors of a 3x3 matrix.
    - **Input Data:**
      ```
      A = np.array([[3, -1, 2], [4, 1, 0], [2, -1, 1]])
      ```
    - **Expected Output:** The eigenvalues and eigenvectors.
    - **Hint:** Use `numpy.linalg.eig()`.

