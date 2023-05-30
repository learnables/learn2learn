# Demystifying Task-Transforms

Written by [Varad Pimpalkhute](https://nightlessbaron.github.io/) on 02/18/2022.

Notebook of this tutorial is available on [Colab Notebook](https://colab.research.google.com/drive/1x7Hf59ulOFXZkkaw_oXXSFHU5hqLp_jL?usp=sharing).

In this tutorial, we will explore in depth one of the core utilities [learn2learn](https://github.com/learnables/learn2learn) library provides - Task Generators. 

## Overview

*   We will first discuss the motivation behind generating tasks. *(Those familiar with meta-learning can skip this section.)*
*   Next, we will have a high-level overview of the overall pipeline used for generating tasks using `learn2learn`.
*   `MetaDataset` is used fast indexing, and accelerates the process of generating few-shot learning tasks. `UnionMetaDataset` and `FilteredMetaDataset` are extensions of `MetaDataset` that can further provide customised utility. `UnionMetaDataset` builds up on `MetaDataset` to construct a union of multiple input datasets, and `FilteredMetaDataset` takes in a `MetaDataset` and filters it to include only the required labels.
*   `Taskset` is the core module that generates tasks from input dataset. Tasks are lazily sampled upon indexing or calling `.sample()` method.
*   Lastly, we study different `task transforms` defined in `learn2learn` that modifies the input data such that a customised `task` is generated. 

## Motivation for generating tasks

#### What is a task?

Let's first start with understanding what is a task. The definition of a task varies from one application to other, but in context of few-shot learning, a task is a *supervised-learning* approach (e.g., classification, regression) trained over a collection of _datapoints_ (images, in context of vision) that are sampled from the same distribution.

> *For example*, a task may consist of 5 images from 5 different classes - flower, cup, bird, fruit, clock (say, 1 image per class),  all sampled from the same distribution. Now, the objective of the task might be to classify the images present at test time amongst the five classes - that is, minimize over the loss function.


<p style="text-align:center;">
<img src="../few-shot.png" style="width:600; vertical-align: middle;"/>
    <div class="caption">
    Few-Shot Classification Tasks. Image source: <a href="https://meta-learning.fastforwardlabs.com/">Cloudera Fast Forward Report on Meta-Learning</a>.
    </div>
</p>



---


#### How is a task used in context of meta-learning?

Meta-learning used in the context of few-shot learning paradigm trains over different tasks (each task consists of limited number of samples) over multiple iterations of training. For example, gradient-based meta-learners learn a model initialization prior such that the model converges to the global minima on unseen tasks (tasks that were not encountered during training) using few samples/datapoints. 


---


#### How is a task generated?

In layman's terms, few-shot classification experiment is set up as a N-wayed K-shot problem. Meaning, the model needs to learn how to classify an input task over N different classes given K examples per class during training. Thus, we need to generate 'M' such tasks for training, and inferencing the meta-learner.

**Summarizing**, we require to:

1. Iterate over classes and their respective samples present in the dataset rapidly in order to generate a task.
2. Write a protocol that generates a task that adhers to the few-shot paradigm *(that is, `N-way K-shot` problem)*.
3. Incorporate additional transforms *(say, augmentation of data)*.
4. Generate `M` randomly sampled tasks for training and inferencing.


Given any input dataset, `learn2learn` makes it easy for generating custom tasks depending on the user's usecase.


---


~~~python
# Import modules that would be used later.
import os, random, pickle, itertools, copy
import learn2learn as l2l
import torchvision as tv
from PIL.Image import LANCZOS
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.vision.transforms import RandomClassRotation
from collections import defaultdict
~~~

## Overview of pipeline for generating tasks

Given any input dataset, `learn2learn` makes it easy for generating custom tasks depending on the user's usecase. A high-level overall pipeline is shown in the diagram below:

The dataset consists of 100 different classes, having 5 samples per class. The objective is to generate N-wayed K-shot task (say, 3-way 2-shot task from the given dataset.)


<p style="text-align:center;">
<img src="../l2l-task-transform.png" style="width:600; vertical-align: middle;"/>
    <div class="caption">
    Pipeline for generating tasks in learn2learn.
    </div>
</p>

The below code snippet shows to generate customised tasks using any input custom using `learn2learn`.

~~~python
# 1. Apply transforms on input data if any (eg., Resizing, converting arrays to tensors)
data_transform = tv.transforms.Compose([tv.transforms.Resize((28, 28), interpolation=LANCZOS), tv.transforms.ToTensor()]) 

# 2. Download the dataset, and apply the above transforms on each of the samples
dataset = l2l.vision.datasets.FullOmniglot(root='~\data', transform=data_transform, download=True) # Load your custom dataset

# 3. Wrap the dataset using MetaDataset to enable fast indexing
omniglot = l2l.data.MetaDataset(dataset) # Use MetaDataset to do fast indexing of samples

# 4. Specify transforms to be used for generating tasks
transforms = [
                    NWays(omniglot, 5),  # Samples N random classes per task (here, N = 5)
                    KShots(omniglot, 1), # Samples K samples per class from the above N classes (here, K = 1) 
                    LoadData(omniglot), # Loads a sample from the dataset
                    RemapLabels(omniglot), # Remaps labels starting from zero
                    ConsecutiveLabels(omniglot), # Re-orders samples s.t. they are sorted in consecutive order 
                    RandomClassRotation(omniglot, [0, 90, 180, 270]) # Randomly rotate sample over x degrees (only for vision tasks)
                    ]

# 5. Generate set of tasks using the dataset, and transforms
taskset = l2l.data.Taskset(dataset=omniglot, task_transforms=transforms, num_tasks=10) # Creates sets of tasks from the dataset 

# Now sample a task from the taskset
X, y = taskset.sample()
# or, you can also sample this way:
for task in taskset:
    X, y = task
print(X.shape)
~~~


> (out): torch.Size([5, 1, 28, 28])

And that's it! You have now generated one task randomly sampled from the omniglot dataset distribution. For generating `M` tasks, you will need to sample the taskset `M` times.

For the rest of the tutorial, we will inspect each of the modules present in the above code, and discuss a few general strategies that can be used while generating tasks efficiently.

## MetaDataset - A wrapper for fast indexing of samples.

At a high level, `MetaDataset` is a wrapper that enables fast indexing of samples of a given class in a dataset. The motivation behind building is to decrease the \(\mathcal{O}(n)\) time to \(\mathcal{O}(1)\) everytime we iterate over a dataset to generate tasks. Naturally, the time saved increases as the dataset size keeps on increasing. 

>**Note** : The input dataset needs to be iterable.

`learn2learn` does this by maintaining two dictionaries for each classification dataset:
1. `labels_to_indices`: A dictionary that maintains labels of classes as keys, and the corresponding indices of samples within the class in form of list as values.
2. `indices_to_labels`: As the name suggests, a dictionary is formed with indices of samples as key, and their corresponding class labels as value.

This feature comes in handy while generating tasks. For example, if you are sampling a task having `N` classes *(say, N=5)*, then using `labels_to_indices` dictionary to identify all the samples belonging to this set of 5 classes *(\(\mathcal{O}(c)\))* will be much more faster than iterating all the samples *(\(\mathcal{O}(n)\))*.

~~~python
from collections import defaultdict

labels_to_indices = defaultdict(list) # each label will store list of sample indices
indices_to_labels = defaultdict(int) # each index will store corresponding sample's label

for i in range(len(dataset)): # iterate over all samples
    
    label = dataset[i][1] # load label
    if hasattr(label, 'item'):
        label = dataset[i][1].item() # if label is a Tensor, then take get the scalar value

    labels_to_indices[label].append(i) # append sample's index to the given class
    indices_to_labels[i] = label # assign label to the given index
~~~


---


> Any one of the two dictionaries can also be optionally passed as an argument upon instantiation, and the other dictionary is built using this dictionary  (See Line [81](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/meta_dataset.pyx#L81) - Line [90](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/meta_dataset.pyx#L90) on GitHub.)


---


#### Bookkeeping

`learn2learn` also provides another utility in the form of an attribute `_bookkeeping_path`. If the input dataset has the given attribute, then the built attributes *(namely, the two dictionaries, and list of labels)* will be cached on disk for latter use. It is recommended to use this utility if:

1. If the dataset size is large, as it can take hours for instantiating it the first time.
2. If you are going to use the dataset again for training. *(Iterating over all the samples will be much slower than loading it from disk)*

To use the bookkeeping utility, while loading your custom dataset, you will need to add an additional attribute to it. 

*For example, we add `_bookkeeping_path` attribute while generating [FC100](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/fc100.py) dataset as follows:* 

`self._bookkeeping_path = os.path.join(self.root, 'fc100-bookkeeping-' + mode + '.pkl')`

where, mode is either *train, validation,* or *test* (depends on how you are defining your dataset. It's also possible that you are loading the entire dataset, and then creating train-valid-test splits. In that case, you can remove the mode variable)

~~~python
import pickle

# The variables can be loaded from the saved .pkl file
with open('~\data/omniglot-bookkeeping.pkl', 'rb') as f:
    _bookkeeping = pickle.load(f)

# In _bookkeeping, the three attributes are saved in the form of a dictionary
labels_to_indices = _bookkeeping['labels_to_indices']
indices_to_labels = _bookkeeping['indices_to_labels']
labels = _bookkeeping['labels']

print('labels_to_indices: label as keys, and list of sample indices as values')
print(labels_to_indices)
print('\n')
print('indices_to_labels: index as key, and corresponding label as value')
print(indices_to_labels)
~~~

> (out): labels_to_indices: label as keys, and list of sample indices as values
> defaultdict(<class 'list'>, {0: [0, 1, 2, 3, 4, ..., 16, 17, 18, 19], 1: [20, 21, 22, 23, 24, 25, ..., 28, 29], ..., 1622: [32440, 32441, 32442, 32443, 32444, ..., 32458, 32459]})
>
> indices_to_labels: index as key, and corresponding label as value
> defaultdict(<class 'int'>, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6, ..., 32458: 1622, 32459: 1622})

So far, we understood the motivation for using `MetaDataset`. In the next sections, we will discuss exactly how the dictionaries generated using `MetaDataset` are used for creating a task.

### UnionMetaDataset - A wrapper for multiple datasets

`UnionMetaDataset` is an extension of `MetaDataset`, and it is used to merge multiple datasets into one. This is useful when you want to sample heterogenous tasks - tasks in a metabatch being from different distributions.

`learn2learn` implements it by simply remapping the labels of the dataset in a consecutive order. For example, say you have two datasets: \(\mathcal{D}_1\): {1600 labels, 32000 samples} and \(\mathcal{D}_2\): {400 labels, 12000 samples}. After wrapping the datasets using `UnionMetaDataset` we will get a MetaDataset that will have 2000 labels and 44000 samples, where the initial 1600 labels (0, 1, 2, ..., 1599) will be from \(\mathcal{D}_1\) and (1600, 1601, ..., 1999) labels will be from \(\mathcal{D}_2\). Same is the case for the indices of the union.

> A list of datasets is fed as input the `UnionMetaDataset`

Below code shows how a high level implementation of the wrapper:

~~~python
dataset_omniglot = l2l.vision.datasets.FullOmniglot(root='~\data', transform=data_transform, download=True)
dataset_omniglot = l2l.data.MetaDataset(dataset_omniglot) # If not wrapped, UnionMetaDataset takes care of it.
dataset_fc100 = l2l.vision.datasets.FC100(root='~\data', transform=data_transform, download=True)
dataset_fc100 = l2l.data.MetaDataset(dataset_fc100)
dataset_cifarfs = l2l.vision.datasets.CIFARFS(root='~\data', transform=data_transform, download=True)
dataset_cifarfs = l2l.data.MetaDataset(dataset_cifarfs)
datasets = [dataset_omniglot, dataset_fc100, dataset_cifarfs]
union = l2l.data.UnionMetaDataset(datasets) # a list of datasets is fed as argument
~~~

~~~python
if len(union.labels_to_indices) == len(dataset_omniglot.labels_to_indices) + len(dataset_fc100.labels_to_indices) + len(dataset_cifarfs.labels_to_indices):
    print('Union was successful')
else:
    print('Something is wrong!')
~~~

> (out): Union was successful

To retrieve a data sample using index, `UnionMetaDataset` iterates over all the individual datasets as follows:

~~~python
# For sake of example, consider item = 1690
# Union of 2 datasets having [1600, 20] and [600, 10] samples & labels respectively
# Thus, item is in D2 having index 89 and say has label as 3
def get_item(item, datasets, union):
    ds_count = 0
    for dataset in datasets: # [D1, D2] (D1 -> 1600 samples, D2 -> 600 samples)
        if ds_count + len(dataset) > item: # For D1, condition fails
            data = list(dataset[item - ds_count]) # searches in the D2's index space
            data[1] = union.indices_to_labels[item] # changes label from 3 to 22 (20 + 3 - 1)
            return data
        ds_count += len(dataset) # Now, ds_count = 1600

get_item(62000, datasets, union)[1] # Returns modified label 
~~~

> (out): 1670

### FilteredMetaDataset - Filter out unwanted labels

`FilteredMetaDataset` is a wrapper that takes in a `MetaDataset` and filters it to only include a subset of the desired labels.

The labels included are not remapped, and the label value from the original dataset is retained.

~~~python
toy_omniglot = l2l.vision.datasets.FullOmniglot(root='~\data', transform=data_transform, download=True)
toy_omniglot = l2l.data.MetaDataset(toy_omniglot)
filtered = l2l.data.FilteredMetaDataset(toy_omniglot, [4, 8, 2, 1, 9])
print('Original Labels:', len(toy_omniglot.labels))
print('Filtered Labels:', len(filtered.labels))
~~~

## Taskset - Core module

#### Introduction

This is one of the core module of `learn2learn` that is used to generate a task from a given input dataset. It takes `dataset`, and list of `task transformations` as arguments. The task transformation basically define the kind of tasks that will be generated from the dataset. (For example, `KShots` transform limits the number of samples per class in a task to `K` samples per class.) 

> If there are no task transforms, then the task consists of all the samples in the entire dataset.

Another argument that `Taskset` takes as input is `num_tasks` *(an integer value)*. The value is set depending on how many tasks the user wants to generate. By default, it is kept as `-1`, meaning infinite number of tasks will be generated, and a new task is generated on sampling. In the former case, the descriptions of the task will be cached in a dictionary such that if a given task is called again, the description can be loaded instantly rather than generating it once again.

#### What is a task description?

A `task_description` is a list of `DataDescription` objects with two attributes: `index`, and `transforms`. `Index` corresponds to the index of a sample in the dataset, and `transforms` is a list of transformations that will be applied to the sample.

~~~python
# Transforms: [NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels, RandomClassRotation]

description = None
for transform in transforms:
    description = transform(description)
    print(len(description), description)

# Initially, there are all samples present in the dataset
# NWays chooses N classes (5 in our case), and samples all the datapoints belonging to only these N classes -> 100 samples (as each class has 20 samples)
# Next, KShot chooses K samples (1 samples per class in our case) from each of these N classes -> thus reducing the total samples in the task_description to 5
# And rest of the task transforms do other specicial transformations on the samples without changing the number of samples present in the description 
~~~

#### How is a task generated?

**STEP 1**

An index between `[0, num_tasks)` is randomly generated.\
(If `num_tasks = -1`, then index is always 0.)

~~~python
import random

def sample(num_tasks):
    i = random.randint(0, num_tasks - 1)
    return i

sample(20000)
~~~

**STEP 2**

There are two possible methods for generating `task_description`:

1. If there's a cached description for the given index, the `task_description` is assigned the cached description. 

2. Otherwise, each transform takes the description returned by the previous transform as argument, and in turn returns a new description. 

> The above only holds true when `num_tasks != -1`, for `num_tasks = -1`, new description is computed every time.

> **NOTE -** It is to be noted `task_description` and `data_description` are general methods and can be used for any type of task, be it a classification task, regression task, or even a timeseries task.

Below code discusses both the methods.

~~~python
# Method 1: Cached Description
# If there's a cached description for the given index, the `task_description` is assigned the cached description. 

sampled_descriptions = {} # a dictionary for caching descriptions
if i not in sampled_descriptions: # i is the index of task between [0, num_tasks]
    sampled_descriptions[i] = 'Call Method 2' # call sample_task_description()
task_description = sampled_descriptions[i]
~~~

~~~python
# Method 2: If method 1 fails, or num_tasks = -1
# 2.Each transform takes the description returned by the previous transform as argument, and in turn returns a new description.

def sample_task_description(self):
    #  Samples a new task description.
    description = None # initialize description as None at the start
    if callable(self.task_transforms):
        return self.task_transforms(description)
    for transform in self.task_transforms: # iterate on the transfroms list [NWays, Kshots, LoadData, ...]
        description = transform(description) # use the description generated by the  previous transform for the current transform
    return description # A description modified by all the transforms present in the list.
~~~

**STEP 3**

Once a `task_description` is retrieved/generated, task is generated by applying the list of transformations present in each of the `DataDescription` objects in the task description list. 

> The transformations mentioned above are different from `task_transforms` (`task_transforms` examples: `NWays`, `KShots`, `LoadData`, etc.) 

All the data samples generated in the list are accumulated and collated using `task_collate`. (by default, `task_collate` is assigned `collate.default_collate`)

`DataDescription` object has two attributes:
`index` of the sample and any `transforms` that need to be applied on the sample. 

~~~python
def get_task(self, task_description):
    #  Given a task description, creates the corresponding batch of data.
    all_data = []
    for data_description in task_description: # iterate over all the samples in task description
        data = data_description.index # loads index of the sample present in the original dataset
        for transform in data_description.transforms:
            # There are two task_transforms (LoadData and RemapLabels) that apply transforms on the data
            # as of now in the learn2learn library.  
            data = transform(data) # applies transform sequentially on the sample data
        all_data.append(data)
    return self.task_collate(all_data) # by default makes use of collate.default_colllate
~~~

We will be discussing more about the `data_description.transforms` in the next section, after which there will be more clarity on exactly how the above snippet modifies the data.

#### A few general tips

1. If you have not wrapped the dataset with `MetaDataset` or its variants, the function will automatically instantiate `MetaDataset` wrapper.

2. If you are not sure how many tasks you want to generate, use `num_tasks = -1`.

3. If `num_tasks = N`, and you are sampling `M` tasks where `M > N`, then `M - N` tasks will definitely be repeated. In case you want to avoid it, make sure `N >= M`.

4. Given a list of task transformations, the transformations are applied in the order they are listed. (Task generated using \([T_1, T_2]\) transforms might be different from that generated using \([T_2, T_1]\).

5. A `task` is lazily sampled upon indexing, or using `.sample()`. `.sample()` is equivalent to indexing, just that before indexing, it randomly generates an index to be indexed.

6. When using `KShots` transform, query twice the samples required for training. The queried samples will need to be split in half, for training, and evaluation. `learn2learn` provides a nice utility called `partition_task()` to partition the data in support and query sets. Check [this](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py#L87) to know more about it. A quick use case:
~~~python
# k-shot learning scenario
(adapt_data, adapt_labels), (eval_data, eval_labels) = partition_task(data, labels, shots=k)
~~~

7. `task_description` and `data_description` are general methods and can be used for any type of task, be it a classification task, regression task, or even a timeseries task.

In the next section, we will examine how the `task_transforms` exactly modify the input dataset to generate a task.

~~~python
class DataDescription:
    def __init__(self, index):
        self.index = index
        self.transforms = []
~~~

## Task Tranforms - Modifying the input dataset

Task transforms are a set of transformations that decide on what kind of a task is generated at the end. We will quickly go over some of the transforms defined in `learn2learn`, and examine how they are used.

To reiterate, a DataDescripton is a class that has two attributes: index, and transforms. Index stores the index of the data sample, and transforms stores list of transforms if there are any (transforms is different from task transforms). In layman's words, it stores indices of samples in the dataset.

> Only `LoadData` and `RemapLabels` add transforms in the list of transform attribute in `DataDescription` object.

#### High-Level Interface

Each of the task transform classes is inherited from `TaskTranform` class. All of them have a common skeleton in the form of three methods namely: `__init__()`, `__call__()` and `new_task()`.

We will now discuss what each of these methods do in general.

**`__init__()` Method**

Initializes the newly created object, in the transform, while also inheriting some arguments such as the dataset from the parent class. Objects / variables that needed to be instantiated only again are defined here.

**`__call__()` Method**

It's a callable method, and is used as a function to write the `task_transform` specific functionality. Objects / variables that keep on changing are defined here.

**`new_task()` Method**

If the `task_description` is empty (that is, `None`), then this method is called. This method loads all the samples present in the dataset to the `task_description`. For instance, check the code below. It loads all the samples present in the dataset to the `task_description` 

~~~python
def new_task(self):
    n = len(self.dataset)
    task_description = [None] * n # generate an empty matrix
    for i in range(n):
        task_description[i] = DataDescription(i) # Add samples wrapped around DataDescription to the list.
    return task_description
~~~

#### A) FusedNWaysKShots

Efficient implementation of `KShots`, `NWays`, and `FilterLabels` transforms. We will be discussing each of the individual transforms in the subsequent sections. 

If you are planning to make use of more than 1 or these transforms, it is recommended to make use of `FusedNWaysKshots` transform instead of using each of them individually. 

#### B) NWays

Keeps samples from `N` random labels present in the task description. `NWays` iterate over the current task description to generate a new description as follows:


*   If no `task_description` is available, `NWays` randomly samples `N` labels, and adds all the samples in these `N` random labels using `labels_to_indices` dictionary.
*   Else, using `indices_to_labels` dictionary, it first identifies the unique labels present in the description. Next, it randomly samples `N` labels from the set of classes.
*   Lastly, it iterates over all the indices present in the description. If the `index` belongs to the set of these `N` random labels, the sample is added in the new `task_description`.

~~~python
# Step 1 : When no task description is available, sample pool is entire dataset
classes = random.sample(labels, k=5) # Randomly sample K classes that will be used for generating the task
example = []
for cl in classes: # add only samples belonging to these classes 
    for idx in labels_to_indices[cl]:
        # Adds the sample index to the task description wrapped in DataDescription object
        # task_description.append(DataDescription(idx))
        # For sake of explaination I am adding the next step
        example.append(idx)
print(example)
print("Number of samples:", len(example)) # should be 100 as each class has 20 samples in omniglot
~~~

> (out): [14380, 14381, 14382, ..., 31196, 31197, 31198, 31199]
> Number of samples: 100

~~~python
# Step 2 : If there's an existing task description, identify unique labels present in the description
def unique_labels():
    set_classes = set() # to remove repeated entries
    for dd in task_description:
        set_classes.add(indices_to_labels[dd.index]) # adds label of each sample index present in the description 
    classes = list(set_classes)
    return classes # returns unique list of classes

# Step 3 : Sample indices belonging the list of randomly choosen classes
def n_samples():
    result = []
    classes = random.sample(unique_labels(), k=5)
    for dd in task_description:
        if indices_to_labels[dd.index] in classes:
            result.append(dd) # adds all the indices belonging to the 5 randomly choosen classes 
    return result # return new task description
~~~

#### C) KShots

It samples `K` samples per label from all the labels present in the `task_desription`. Similar to `NWays`, `KShots` iterate over the samples present in the current `task_description` to generate a new one:

*   If `task_description` is `None`, load all the samples present in the dataset.
*   Else, generate a `class_to_data` dictionary that stores label as key and corresponding samples as value.
*   Lastly, `K` samples are sampled from each of the classes either with or without replacement.


~~~python
# Step 1 : Task description is None
# Load all samples in dataset
task_description = [None] * len(dataset)
for i in range(len(dataset)):
    task_description[i] = DataDescription(i)

# Step 2 : Create a dictionary that stores labels and their corresponding samples
class_to_data = defaultdict(list)
for dd in task_description:
    cls = indices_to_labels[dd.index]
    class_to_data[cls].append(dd)

# Step 3 : Sample K datapoints from each class with or without replacement
# if with replacement, use
def sampler(x, k):
    return [copy.deepcopy(dd) for dd in random.choices(x, k=k)]
# else use
sampler = random.sample
# Lastly, sample the datapoints
len(list(itertools.chain(*[sampler(dds, k=2) for dds in class_to_data.values()]))) # here, we are sampling 2 datapoints per class.
# There are 1623 classes having 20 samples per class -> 32460 datapoints
# Thus, if we sample 2 datapoints per class, that will leave us with 3246 datapoints
~~~

> (out): 3246

#### D) LoadData

Loads a sample from the dataset given its index. Does so by appending a transform `lambda x: self.dataset[x]` to `transforms` attribute present in DataDescription for each sample.

~~~python
# Loads the data using transforms which will be used when calling get_task() in task_dataset.pyx
for data_description in task_description:
    data_description.transforms.append(lambda x: dataset[x])
~~~


---


> The above three task transforms are the main transforms that are usually used when generating few-shot learning tasks. These transforms can be used in any other.


---


#### E) FilterLabels

It's a simple transform that removes any unwanted labels from the `task_description`. In addition to the dataset, it takes a list of labels that need to be included as an argument.

*   It first generates filtered indices that keep a track on all the indices of the samples from the input labels.
*   Next, it iterates over all the indices in the task description, and filters them out if they don't belong to the filtered indices.

> If you are using FilterLabels transform, it is recommended to use it before NWays, and KShots transforms.

~~~python
# Step 1 : Generate filtered indices
import array

filtered_labels = [1, 3, 5, 7, 9]
filtered_indices = array.array('i', [0] * len(dataset))
for i in range(len(dataset)):
    # will generate a sparse list with labels for the samples with filtered labels
    filtered_indices[i] = int(indices_to_labels[i] in filtered_labels) 

# Step 2 : Filter out descriptions that don't belong to the filtered indices
result = []
for dd in task_description:
    if filtered_indices[dd.index]: # if index value is non-zero
        result.append(dd)
print(result)
print(len(result)) # 20 samples for 5 classes each -> 100 samples
~~~

#### F) ConsecutiveLabels

The transform re-orders the samples present in the `task_description` according to the label order consecutively. If you are using `RemapLabels` transform and keeping `shuffle=True`, it is recommended to keep `ConsecutiveLabels` tranform after `RemapLabels`, otherwise, while they will be homogeneously clustered, the ordering would be random. If you are using `ConsecutiveLabels` transform before `RemapLabels`, and want ordered set of labels, then keep `shuffle=False`.

~~~python
# How consecutive labels transform is implemented
pairs = [(dd, indices_to_labels[dd.index]) for dd in task_description]
pairs = sorted(pairs, key=lambda x: x[1]) # sort by labels x[0] : index, x[1] : label
print([p[0] for p in pairs])

# Example demonstration
toy_dataset = [i for i in range(1000)] # generate a toy dataset
toy_indices_to_labels = {}
for i in toy_dataset:
    toy_indices_to_labels[i] = random.randint(0, 99) # generate a toy indices to labels dictionary

toy_list = [i for i in range(1000)] # generate a toy samples list
toy_task_description = [random.choice(toy_list) for _ in range(10)] # generate a random task description
pairs = [(dd, toy_indices_to_labels[dd]) for dd in toy_task_description]
pairs = sorted(pairs, key=lambda x: x[1]) # sort the pairs list by using the second element (labels) in the tuple

print('\n')
print([p[0] for p in pairs]) # prints index (not ordered)
print([p[1] for p in pairs]) # prints label (ordered list)
~~~

> (out): [271, 702, 756, 319, 948, 840, 843, 741, 89, 413]
> [13, 33, 34, 46, 56, 57, 62, 70, 76, 92]

#### G) RemapLabels

The transform maps the labels of input to `0, 1, ..., N` (given `N` unique set of labels). 

*For example*, if input `task_description` consists of samples from 3 labels namely 71, 14 and 89, then the transform maps the labels to 0, 1 and 2. Compulsorily needs to be present after `LoadData` transform in the transform list, otherwise, will give a `TypeError: int is not iterable`.

The error occurs because `RemapLabels` expects the input to be of iterable form. Thus, unless you load data using `LoadData` prior to it, it will try to iterate over sample `index`, which is an `int`, and not an iterable.

~~~python
import traceback

toy_transforms = [
                NWays(omniglot, 5),  # Samples N random classes per task (here, N=5)
                KShots(omniglot, 2), # Samples K samples per class from the above N classes (here, K=1) 
                RemapLabels(omniglot),
                LoadData(omniglot), # Loads a sample from the dataset
                ConsecutiveLabels(omniglot), # Re-orders samples s.t. they are sorted in consecutive order 
                RandomClassRotation(omniglot, [0, 90, 180, 270]) # Randomly rotate sample over x degrees (only for vision tasks)
                ]
toy_taskset = l2l.data.Taskset(omniglot, toy_transforms, num_tasks=20000)
try:
    print(len(toy_taskset.sample())) # Expected error as RemapLabels is used before LoadData
except TypeError:
    print(traceback.format_exc())
~~~

~~~shell
(output):
Traceback (most recent call last):
  File "<ipython-input-27-4c0558e6745b>", line 13, in <module>
    print(len(toy_taskset.sample())) # Expected error as RemapLabels is used before LoadData
  File "learn2learn/data/task_dataset.pyx", line 158, in learn2learn.data.task_dataset.CythonTaskset.sample
  File "learn2learn/data/task_dataset.pyx", line 173, in learn2learn.data.task_dataset.CythonTaskset.__getitem__
  File "learn2learn/data/task_dataset.pyx", line 142, in learn2learn.data.task_dataset.CythonTaskset.get_task
  File "learn2learn/data/transforms.pyx", line 201, in learn2learn.data.transforms.RemapLabels.remap
TypeError: 'int' object is not iterable
~~~

## Conclusion

Thus, we studied how `learn2learn` simplifies the process of generating few-shot learning tasks. For more details, have a look at:

1.   [Official Documentation](http://learn2learn.net/docs/learn2learn.data/)
2.   [Module Scripts - GitHub](https://github.com/learnables/learn2learn/tree/master/learn2learn/data)

`learn2learn` provides benchmarks for some of the commonly used computer vision datasets such as `omniglot`, `fc100`, `mini-imagenet`, `cirfarfs` and `tiered-imagenet`. The benchmarks are available at [this](https://github.com/learnables/learn2learn/tree/master/learn2learn/vision/benchmarks) link. 

They are  very easy to use, and can be used as follows:
~~~python
# data_name: 'omniglot', 'fc100', 'cifarfs', 'mini-imagenet', 'tiered-imagenet'
# N: No of ways, K: No of shots
tasksets = l2l.vision.benchmarks.get_tasksets(data_name, train_ways=N, train_samples=2*K, test_ways=N, test_samples=2*K, num_tasks=-1, root='~/data')
X1, y1 = tasksets.train.sample()
X2, y2 = tasksets.validation.sample()
X3, y3 = tasksets.test.sample()
~~~

If you have any other queries - feel free to ask questions on the library's [slack](http://slack.learn2learn.net/) channel, or open an issue [here](https://github.com/learnables/learn2learn/issues).

Thank you! 
