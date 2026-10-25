from capymoa.base import Classifier
from moa.classifiers.lazy.neighboursearch import KDTree
from moa.classifiers.lazy.neighboursearch import LinearNNSearch
from com.yahoo.labs.samoa.instances import Instances
import math
import random

##############################################################
####################### HELPER WINDOW ########################
##############################################################
class newWindow():
    def __init__(self, schema, instances=None):
        self.window = Instances(schema.get_moa_header(), 0)
        self.schema = schema
        if instances is not None:
            for i in range(instances.numInstances()):
                self.window.add(instances.get(i))
    
    def add_instance(self, instance):
        self.window.add(instance.java_instance.getData())

    def add_window(self, nwindow):
        rw = nwindow.get_window()
        for i in range(nwindow.size()):
            self.window.add(rw.get(i))

    def remove_instance(self, position=0):
        self.window.delete(position)

    def get_instance(self, position=0):
        return self.window.get(position)

    def get_window(self):
        return self.window
    
    def get_schema(self):
        return self.schema
    
    def size(self):
        return self.window.numInstances()
    
##############################################################
###################### HELPER NNSEARCH #######################
##############################################################
class SearchNeighbours():
    def __init__(self, search_method):
        assert search_method in ["LinearNNSearch", "KDTree"]
        self.search_method_str = search_method
        if self.search_method_str == "LinearNNSearch":
            self.search_method = LinearNNSearch()
        else:
            self.search_method = KDTree()

    def do_search(self, window, instance, arg_k):
        self.search_method.setInstances(window.get_window())
        k = min(arg_k, window.size())
        neighbors = self.search_method.kNearestNeighbours(instance.java_instance.getData(), k)
        return newWindow(window.get_schema(), neighbors)
    
##############################################################
###################### HELPER DISTANCE #######################
##############################################################
def distanceMath(instance1, instance2):
	if 'LabeledInstance' in type(instance1).__name__:
		inst1 = instance1.java_instance.getData().toDoubleArray()
	if 'LabeledInstance' in type(instance2).__name__:
		inst2 = instance2.java_instance.getData().toDoubleArray()
	elif 'InstanceImpl' in type(instance2).__name__:
		inst2 = instance2.toDoubleArray()
	return math.sqrt(sum([(inst1[i] - inst2[i])**2 for i in range(len(inst1)-1)]))
    
##############################################################
######################## KNN CLASSES #########################
##############################################################
class KNNBase(Classifier):
    def __init__(self, schema, k=3, window_size=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.window_size = window_size
        self.search = SearchNeighbours(search_method)
        self.window = newWindow(schema)

    def __str__(self):
        return "Base kNN"

    def train(self, instance):
        if self.window.size() >= self.window_size:
            self.window.remove_instance()
        self.window.add_instance(instance)
    
    def predict(self, instance):
        """
        Returns the k-NN predictions for a given instance.
        Uses mean/median for regression or majority vote for classification.
        """
        try:
            if self.window and self.window.size() > 0:
                # Find k nearest neighbors
                neighbors = self.search.do_search(self.window, instance, self.k)
                # Vote for the most common class among the k closest neighbors
                votes = [0] * (self.schema.get_num_classes())
                for i in range(neighbors.size()):
                    class_idx = int(neighbors.get_instance(i).classValue())
                    votes[class_idx] += 1
                return votes.index(max(votes))

        except Exception as e:
            print("Error: kNN search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass

##############################################################
################### KNN-CW (Per Class Window) ################
##############################################################
class KNNClassWindows(Classifier):
    def __init__(self, schema, k=3, w=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.w = w
        self.search = SearchNeighbours(search_method)
        self.class_window = {}  # label -> newWindow

    def __str__(self):
        return "KNN-cw"

    def train(self, instance):
        # Get class label of the incoming instance
        label = int(instance.java_instance.getData().classValue())

        # If this is a new class, create a new window for it
        if label not in self.class_window:
            self.class_window[label] = newWindow(self.schema)

        # Add instance to the corresponding class window
        self.class_window[label].add_instance(instance)

        # Enforce per-class window size (FIFO: remove oldest)
        if self.class_window[label].size() > self.w:
            self.class_window[label].remove_instance(0)

    def predict(self, instance):
        """
        Two-stage k-NN prediction with per-class windows:
        1) Find k nearest neighbours in each class window
        2) Merge candidates, find global k nearest, majority vote
        """
        try:
            # Stage 1: collect k nearest from each class window
            all_candidates = newWindow(self.schema)

            for label, cw in self.class_window.items():
                if cw.size() > 0:
                    local_k = min(self.k, cw.size())
                    neighbors = self.search.do_search(cw, instance, local_k)
                    all_candidates.add_window(neighbors)

            if all_candidates.size() == 0:
                return 0

            # Stage 2: find global k nearest from all candidates
            final_k = min(self.k, all_candidates.size())
            final_neighbors = self.search.do_search(
                all_candidates, instance, final_k
            )

            # Stage 3: majority vote
            votes = [0] * self.schema.get_num_classes()
            for i in range(final_neighbors.size()):
                class_idx = int(final_neighbors.get_instance(i).classValue())
                votes[class_idx] += 1
            return votes.index(max(votes))

        except Exception as e:
            print("Error: kNN-cw search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass

##############################################################
############## KNN-ASW (Age-Weighted Sampling) ###############
##############################################################
LOG_1_2 = math.log(1.2)  # precompute ln(1.2)
SEED = 523  # fixed seed for the ejection draws in the reported results

class KNNAgeWeighted(Classifier):
    def __init__(self, schema, k=3, w=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.w = w
        self.search = SearchNeighbours(search_method)
        self.window = newWindow(schema)
        self.ages = []  # parallel list: age of each instance in window
        random.seed(SEED)

    def __str__(self):
        return "KNN-asw"

    def train(self, instance):
        # Step 1: Increment age of all existing instances
        self.ages = [a + 1 for a in self.ages]

        # Step 2: Add new instance with age = 1
        self.window.add_instance(instance)
        self.ages.append(1)

        # Step 3: If window exceeds w, eject one instance probabilistically
        if self.window.size() > self.w:
            self._eject_one()

    def _eject_one(self):
        """
        Eject one instance with probability proportional to 1.2^age.
        Uses log-sum-exp trick to avoid float overflow:
          log_weight_i = age_i * ln(1.2)
          P(i) = exp(log_weight_i - max_log) / sum(exp(log_weight_j - max_log))
        """
        n = self.window.size()

        # Compute log-weights
        log_weights = [self.ages[i] * LOG_1_2 for i in range(n)]
        max_log = max(log_weights)

        # Compute normalized weights via log-sum-exp
        weights = [math.exp(lw - max_log) for lw in log_weights]
        total = sum(weights)

        # Build cumulative distribution and sample
        r = random.random() * total
        cumulative = 0.0
        eject_idx = n - 1  # fallback
        for i in range(n):
            cumulative += weights[i]
            if cumulative >= r:
                eject_idx = i
                break

        # Remove the selected instance
        self.window.remove_instance(eject_idx)
        self.ages.pop(eject_idx)

    def predict(self, instance):
        """
        Standard kNN prediction: find k nearest neighbours, majority vote.
        """
        try:
            if self.window and self.window.size() > 0:
                neighbors = self.search.do_search(self.window, instance, self.k)
                votes = [0] * self.schema.get_num_classes()
                for i in range(neighbors.size()):
                    class_idx = int(neighbors.get_instance(i).classValue())
                    votes[class_idx] += 1
                return votes.index(max(votes))
        except Exception as e:
            print("Error: kNN-asw search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass

##############################################################
################### ORIGINAL BASE3 (keep) ####################
##############################################################
class KNNBase3(Classifier):
    def __init__(self, schema, k=3, window_size=1000, search_method="LinearNNSearch"):
        super().__init__(schema)
        self.k = k
        self.window_size = window_size
        self.search = SearchNeighbours(search_method)
        self.window = newWindow(schema)

    def __str__(self):
        return "Base3 kNN"

    def train(self, instance):
        if self.window.size() >= self.window_size:
            self.window.remove_instance()
        self.window.add_instance(instance)
    
    def predict(self, instance):
        """
        Returns the k-NN predictions for a given instance.
        Uses mean/median for regression or majority vote for classification.
        """
        try:
            if self.window and self.window.size() > 0:
                # Find k nearest neighbors
                neighbors = self.search.do_search(self.window, instance, self.k)
                # Vote for the most common class among the k closest neighbors
                votes = [0] * (self.schema.get_num_classes())
                for i in range(neighbors.size()):
                    class_idx = int(neighbors.get_instance(i).classValue())
                    votes[class_idx] += 1
                return votes.index(max(votes))

        except Exception as e:
            print("Error: kNN search failed.", e)
            return 0

    def predict_proba(self, instance):
        pass