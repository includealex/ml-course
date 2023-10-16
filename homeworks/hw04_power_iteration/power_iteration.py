import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    rand_vec = np.random.rand(data.shape[0])

    for _ in range(num_steps):
        tmp = (data @ rand_vec)
        rand_vec = tmp / np.linalg.norm(tmp)

    eigenval = 1 / rand_vec.dot(rand_vec) * (rand_vec.dot(data @ rand_vec))

    return  float(eigenval), rand_vec
