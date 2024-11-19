import numpy as np
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from scipy.stats import wishart


"""
Class for product space object
"""
class ProductSpace:
    def __init__(self, signature=[], X=None, y=None, seed=None):
        self.signature = signature
        self.check_signature()
        self.X = X
        self.y = y
        self.seed = seed

    def check_signature(self):
        """Check if signature is valid"""
        if len(self.signature) == 0:
            raise ValueError("Signature is empty")
        for space in self.signature:
            if not isinstance(space, tuple):
                raise ValueError("Signature elements must be tuples")
            if len(space) != 2:
                raise ValueError("Signature tuples must have 2 values")
            if not isinstance(space[0], int) or space[0] <= 0:
                raise ValueError("Dimension must be a positive integer")
            if not isinstance(space[1], (int, float)):
                raise ValueError("Curvature must be an integer or float")

    def print_signature(self):
        """Print the signature of the product space"""
        for space in self.signature:
            if space[1] < 0:
                print(f"H: dim={space[0]}, K={space[1]}")
            elif space[1] > 0:
                print(f"S: dim={space[0]}, K={space[1]}")
            else:
                print(f"E: dim={space[0]}")

    def sample_clusters(self, num_points, num_classes, cov_scale=0.3, centers=None):
        """Generate data from a wrapped normal mixture on the product space"""
        self.X, self.y, self.means = [], [], []
        classes = WrappedNormalMixture(
            num_points=num_points, num_classes=num_classes, seed=self.seed
        ).generate_class_assignments()
        for space in self.signature:
            wnm = WrappedNormalMixture(
                num_points=num_points,
                num_classes=num_classes,
                n_dim=space[0],
                curvature=space[1],
                seed=self.seed,
                cov_scale=cov_scale,
            )
            means = wnm.generate_cluster_means(centers=centers)
            covs = [
                wnm.generate_covariance_matrix(wnm.n_dim, wnm.n_dim + 1, wnm.cov_scale) for _ in range(wnm.num_classes)
            ]
            points = wnm.sample_points(means, covs, classes)
            means /= np.sqrt(wnm.k) if wnm.k != 0.0 else 1.0
            self.X.append(points)
            self.y.append(classes)
            self.means.append(means)
            if wnm.curvature != 0.0:
                assert np.allclose(wnm.manifold.metric.squared_norm(points), 1 / wnm.curvature, rtol=1e-4)

        self.X = np.hstack(self.X)  # (num_points, num_spaces * (num_dims+1) )
        self.y = self.y[0]  # (num_points, )
        self.means = np.hstack(self.means)  # (num_classes, num_dims + 1 )

    def split_data(self, test_size=0.2):
        """Split the data into training and testing sets"""
        n = self.X.shape[0]
        np.random.seed(self.seed)
        test_idx = np.random.choice(n, int(test_size * n), replace=False)
        self.X_train = np.delete(self.X, test_idx, axis=0)
        self.X_test = self.X[test_idx]
        self.y_train = np.delete(self.y, test_idx)
        self.y_test = self.y[test_idx]

    def zero_out_spacelike_dims(self, space_idx):
        """Zero out spacelike dimensions in a given product space component"""
        timelike_dim = sum([space[0] + 1 for space in self.signature[:space_idx]])
        self.X[:, timelike_dim] = 1.0 / np.sqrt(abs(self.signature[space_idx][1]))
        for i in range(self.signature[space_idx][0]):
            self.X[:, timelike_dim + i + 1] = 0.0

    def remove_timelike_dims(self):
        """Remove timelike dimensions from the product space"""
        timelike_dims = [0]
        for i in range(len(self.signature) - 1):
            timelike_dims.append(sum([space[0] + 1 for space in self.signature[: i + 1]]))
        self.X = np.delete(self.X, timelike_dims, axis=1)


'''
Wrapped normal mixture class that generates points from a mixture of Gaussians on
the hyperboloid, Euclidean or hypersphere with defined curvature.
'''
class WrappedNormalMixture:
    def __init__(
        self,
        num_points: int,
        num_classes: int,
        n_dim: int = 2,
        curvature: float = 0.0,
        seed: int = None,
        cov_scale: float = 0.3,
    ):
        self.num_points = num_points
        self.num_classes = num_classes
        self.n_dim = n_dim
        self.curvature = curvature
        self.k = abs(curvature)
        self.curv_sign = 1
        self.seed = seed
        self.cov_scale = cov_scale

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)
        
        # Set manifold based on curvature
        if curvature == 0.0:
            self.manifold = Euclidean(dim=n_dim)
        elif curvature > 0.0:
            self.manifold = Hypersphere(dim=n_dim)
        else:
            self.manifold = Hyperboloid(dim=n_dim)
            self.curv_sign = -1

        # Set origin for hyperboloid and hypersphere
        self.origin = np.array([1.0] + [0.0] * self.n_dim)
        
    
    def generate_cluster_means(self, centers=None):
        '''
        Generate random cluster means or given cluster means on the manifold, adjusted for curvature.
        '''

        if centers is None:
            centers = self.rng.normal(size=(self.num_classes, self.n_dim))
        means = np.concatenate(
            (
                np.zeros(shape=(self.num_classes, 1)),
                centers,
            ),
            axis=1,
        )

        # Adjust for curvature
        means *= np.sqrt(self.k) if self.k != 0.0 else 1.0

        return self.manifold.metric.exp(tangent_vec=means, base_point=self.origin)


    def generate_covariance_matrix(self, dims, deg_freedom, scale):
        '''
        Generate random covariance matrix based on Wishart distribution.
        '''
        scale_matrix = scale * np.eye(dims)
        cov_matrix = wishart.rvs(df=deg_freedom, scale=scale_matrix, random_state=self.rng)

        return cov_matrix


    def generate_class_assignments(self):
        '''
        Generate random class assignments based on uniform class probabilities.
        '''
        probs = self.rng.uniform(size=self.num_classes)
        probs = probs / np.sum(probs)

        return self.rng.choice(self.num_classes, size=self.num_points, p=probs)


    def sample_points(self, means, covs, classes):
        '''
        Generate random samples for each cluster based on the cluster means and covariance matrices.
        '''
        # Generate random vectors on tangent plane for each class
        vecs = np.array([self.rng.multivariate_normal(np.zeros(self.n_dim), covs[c]) for c in classes])
        
        # Adjust for curvature and prepend zeros for ambient space
        vecs *= np.sqrt(self.k) if self.k != 0.0 else 1.0
        vecs = np.column_stack((np.zeros(vecs.shape[0]), vecs))

        # Parallel transport vectors from origin to sampled means on the manifold
        tangent_vecs = self.manifold.metric.parallel_transport(vecs, self.origin, end_point=means[classes])

        # Exponential map to manifold at the class mean
        points = self.manifold.metric.exp(tangent_vec=tangent_vecs, base_point=means[classes])
        
        # Adjust for curvature
        points /= np.sqrt(abs(self.k)) if self.k != 0.0 else 1.0

        return points


    def generate_data(self):
        '''
        Generate Gaussian mixture data.
        '''
        # Generate random class means on the manifold
        means = self.generate_cluster_means()
        
        # Generate covariance matrices for each class
        covs = [self.generate_covariance_matrix(self.n_dim, self.n_dim + 1, self.cov_scale) for _ in range(self.num_classes)]

        # Generate class assignments
        classes = self.generate_class_assignments()
        
        # Sample points from the Gaussian mixture
        points = self.sample_points(means, covs, classes)

        # Readjust means for curvature
        means /= np.sqrt(self.k) if self.k != 0.0 else 1.0

        return points, classes, means
