#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction
# this notebook is adapted from Python Data Science Handbook

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


# ## The test example "HELLO"

# In[2]:


def make_hello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)
    
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]


# Let's call the function and visualize the resulting data in 2D space:

# In[3]:


X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5), marker='.')
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal');


# The output is two dimensional, and consists of points drawn in the shape of the word, "HELLO".
# This data form will help us to see visually what these algorithms are doing.

# ## Multidimensional Scaling (MDS)
# 
# First, Let's rotate 'Hello'

# In[4]:


def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)
    
X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal');


# This tells us that the *x* and *y* values are not necessarily fundamental to the relationships in the data.
# What *is* fundamental, in this case, is the *distance* between each point and the other points in the dataset.
# A common way to represent this is to use a distance matrix: for $N$ points, we construct an $N \times N$ array such that entry $(i, j)$ contains the distance between point $i$ and point $j$.
# Let's use Scikit-Learn's efficient ``pairwise_distances`` function to do this for our original data:

# In[5]:


from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
D.shape


# As promised, for our *N*=1,000 points, we obtain a 1000×1000 matrix, which can be visualized as shown here:

# In[6]:


plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar();


# If we similarly construct a distance matrix for our rotated and translated data, we see that it is the same:

# In[7]:


D2 = pairwise_distances(X2)
np.allclose(D, D2)


# This distance matrix gives us a representation of our data that is invariant to rotations and translations, but the visualization of the matrix above is not entirely intuitive.
# In the representation shown in this figure, we have lost any visible sign of the interesting structure in the data: the "HELLO" that we saw before.
# 
# Further, while computing this distance matrix from the (x, y) coordinates is straightforward, transforming the distances back into *x* and *y* coordinates is rather difficult.
# This is exactly what the multidimensional scaling algorithm aims to do: given a distance matrix between points, it recovers a $D$-dimensional coordinate representation of the data.
# Let's see how it works for our distance matrix, using the ``precomputed`` dissimilarity to specify that we are passing a distance matrix:

# In[8]:


from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');


# The MDS algorithm recovers one of the possible two-dimensional coordinate representations of our data, using *only* the $N\times N$ distance matrix describing the relationship between the data points.

# ## MDS as Manifold Learning
# 
# The usefulness of this becomes more apparent when we consider the fact that distance matrices can be computed from data in *any* dimension.
# So, for example, instead of simply rotating the data in the two-dimensional plane, we can project it into three dimensions using the following function (essentially a three-dimensional generalization of the rotation matrix used earlier):

# In[9]:


def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
    
X3 = random_projection(X, 3)
X3.shape


# Let's visualize these points to see what we're working with:

# In[10]:


from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],
             **colorize)
ax.view_init(azim=70, elev=50)


# We can now ask the ``MDS`` estimator to input this three-dimensional data, compute the distance matrix, and then determine the optimal two-dimensional embedding for this distance matrix.
# The result recovers a representation of the original data:

# In[11]:


model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal');


# This is essentially the goal of a manifold learning estimator: given high-dimensional embedded data, it seeks a low-dimensional representation of the data that preserves certain relationships within the data.
# In the case of MDS, the quantity preserved is the distance between every pair of points.

# ## Nonlinear Embeddings: Where MDS Fails
# 
# Our discussion thus far has considered *linear* embeddings, which essentially consist of rotations, translations, and scalings of data into higher-dimensional spaces.
# Where MDS breaks down is when the embedding is nonlinear—that is, when it goes beyond this simple set of operations.
# Consider the following embedding, which takes the input and contorts it into an "S" shape in three dimensions:

# In[12]:


def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T

XS = make_hello_s_curve(X)


# This is again three-dimensional data, but we can see that the embedding is much more complicated:

# In[13]:


from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],
             **colorize);


# The fundamental relationships between the data points are still there, but this time the data has been transformed in a nonlinear way: it has been wrapped-up into the shape of an "S."
# 
# If we try a simple MDS algorithm on this data, it is not able to "unwrap" this nonlinear embedding, and we lose track of the fundamental relationships in the embedded manifold:

# In[14]:


from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal');


# The best two-dimensional *linear* embeding does not unwrap the S-curve, but instead throws out the original y-axis.

# ## Locally Linear Embedding to unwrap the S-curve

# In[15]:


from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified',
                               eigen_solver='dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15);


# The result remains somewhat distorted compared to our original manifold, but captures the essential relationships in the data!

# Now, let's try isomap on the S-curve

# In[16]:


from sklearn.manifold import Isomap
model = Isomap(n_components=2, n_neighbors=3)
out_iso = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out_iso[:, 0], out_iso[:, 1], **colorize)
#ax.set_ylim(0.15, -0.15);


# Let's try PCA on the S-curve

# In[17]:


from sklearn.decomposition import PCA
model = PCA(n_components=2)
out = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)


# Let's try PCA on the TSNE

# In[18]:


from sklearn.manifold import TSNE
model = TSNE(n_components=2)
out = model.fit_transform(XS)
fig, ax = plt.subplots()
ax.scatter(out_iso[:, 0], out_iso[:, 1], **colorize)


# ## Example: Isomap on Faces

# In[19]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)
faces.data.shape


# We have 2,370 images, each with 2,914 pixels.
# In other words, the images can be thought of as data points in a 2,914-dimensional space!
# 
# Let's quickly visualize several of these images to see what we're working with:

# In[20]:


fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')


# We would like to plot a low-dimensional embedding of the 2,914-dimensional data to learn the fundamental relationships between the images.
# One useful way to start is to compute a PCA, and examine the explained variance ratio, which will give us an idea of how many linear features are required to describe the data:

# In[21]:


from sklearn.decomposition import PCA
model = PCA(n_components=100, svd_solver='randomized').fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance');


# We see that for this data, nearly 100 components are required to preserve 90% of the variance: this tells us that the data is intrinsically very high dimensional—it can't be described linearly with just a few components.
# 
# When this is the case, nonlinear manifold embeddings like LLE and Isomap can be helpful.
# We can compute an Isomap embedding on these faces using the same pattern shown before:

# In[22]:


from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
proj.shape


# The output is a two-dimensional projection of all the input images.
# To get a better idea of what the projection tells us, let's define a function that will output image thumbnails at the locations of the projections:

# In[23]:


from matplotlib import offsetbox

def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)


# Calling this function now, we see the result:

# In[24]:


fig, ax = plt.subplots(figsize=(10, 10))
model_here = Isomap(n_components=2)  # SpectralEmbedding(n_components=2), #PCA(n_components=2, svd_solver='randomized'),
plot_components(faces.data,
                model = model_here,
                images=faces.images[:, ::2, ::2])


# The result is interesting: the first two Isomap dimensions seem to describe global image features: the overall darkness or lightness of the image from left to right, and the general orientation of the face from bottom to top.
# This gives us a nice visual indication of some of the fundamental features in our data.

# ## Example: Visualizing Structure in Digits

# In[26]:


mnist_data=np.load('mnist.data.npy')
mnist_target=np.load('mnist.target.npy')


# This consists of 70,000 images, each with 784 pixels (i.e. the images are 28×28).
# As before, we can take a look at the first few images:

# In[27]:


fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist_data[1250 * i].reshape(28, 28), cmap='gray_r')


# This gives us an idea of the variety of handwriting styles in the dataset.
# 
# Let's compute a manifold learning projection across the data.
# For speed here, we'll only use 1/30 of the data, which is about ~2000 points
# (because of the relatively poor scaling of manifold learning, I find that a few thousand samples is a good number to start with for relatively quick exploration before moving to a full calculation):

# The resulting scatter plot shows some of the relationships between the data points, but is a bit crowded.
# We can gain more insight by looking at just a single number at a time:

# In[28]:


from sklearn.manifold import Isomap

# Choose 1/4 of the "1" digits to project
data = mnist_data[mnist_target == 1][::1]

fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense') # SpectralEmbedding(n_components=2) # PCA(n_components=2, svd_solver='randomized') # 
plot_components(data, model, images=data.reshape((-1, 28, 28)),
                ax=ax, thumb_frac=0.05, cmap='gray_r')


# In[ ]:


# use only 1/30 of the data: full dataset takes a long time!
data = mnist_data[::30]
target = mnist_target[::30]

model = Isomap(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);


# In[ ]:


# use only 1/30 of the data: full dataset takes a long time!
data = mnist_data[::30]
target = mnist_target[::30]

model = PCA(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);


# In[ ]:


# use only 1/30 of the data: full dataset takes a long time!
data = mnist_data[::30]
target = mnist_target[::30]

model = TSNE(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);

