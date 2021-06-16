import csv
import os
import random
import math
import pprint
import pandas as pd


def read_data(csv_path):
    """Read in the training data from a csv file.

    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                        example[k] = v

            examples.append(example)
    return examples


# data['Female'] = [1 if x == "Female" else 0 for x in data['Female']]
# data['Married'] = [1 if x == "Yes" else 0 for x in data['Married']]
# data['Graduate'] = [1 if x == "Graduate" else 0 for x in data['Graduate']]
# data['Self_Employed'] = [1 if x == "Yes" else 0 for x in data['Self_Employed']]

# for keys in data:
#     print(f"{keys} {set(data[keys]) if len(set(data[keys])) < 10 else -1}")

# print(data['Dependents'][7], type(data['Dependents'][7]))


def clean_data(data):
    data = pd.DataFrame(data=data)

    data = data.rename({'Gender': 'Female', 'Education': 'Graduate'}, axis=1)
    for key in data:
        col = data[key]
        # print(key, col[0])
        new_col = []
        if key == "Female":
            for x in col:
                if x == "Female":
                    new_col.append(1)
                elif x == "Male":
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
        if key == "Married":
            for x in col:
                if x == "Yes":
                    new_col.append(1)
                elif x == "No":
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
        if key == "Graduate":
            for x in col:
                if x == "Graduate":
                    new_col.append(1)
                elif x == "Not Graduate":
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
        if key == "Self_Employed":
            for x in col:
                if x == "Yes":
                    new_col.append(1)
                elif x == "No":
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
        if key == "Property_Area":
            for x in col:
                if x == "Urban":
                    new_col.append(2)
                elif x == "Semiurban":
                    new_col.append(1)
                elif x == "Rural":
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
        if key == "Dependents":
            for x in col:
                if x == "3+":
                    new_col.append(3)
                elif x == 2.0:
                    new_col.append(2)
                elif x == 1.0:
                    new_col.append(1)
                elif x == 0.0:
                    new_col.append(0)
                else:
                    new_col.append(x)
            data[key] = new_col
    data = data.to_dict('records')
    return data


def rollback_data(attr, val):
    ret_attr = attr
    ret_val = val
    if attr == "Female":
        ret_attr = "Gender"
        if int(val) == 1:
            ret_val = "Female"
        elif int(val) == 0:
            ret_val = "Male"
    if attr == "Married":
        ret_attr = "Married"
        if int(val) == 1:
            ret_val = "Yes"
        elif int(val) == 0:
            ret_val = "No"
    if attr == "Graduation":
        ret_attr = "Education"
        if int(val) == 1:
            ret_val = "Graduated"
        elif int(val) == 0:
            ret_val = "Not Graduate"
    if attr == "Self_Employed":
        ret_attr = "Self_Employed"
        if int(val) == 1:
            ret_val = "Yes"
        elif int(val) == 0:
            ret_val = "No"
    if attr == "Property_Area":
        ret_attr = "Property_Area"
        if int(val) == 2:
            ret_val = "Urban"
        elif int(val) == 1:
            ret_val = "Semiurban"
        elif int(val) == 0:
            ret_val = "Rural"
    if attr == "Dependents":
        ret_attr = "Dependents"
        if int(val) == 3:
            ret_val = "3+"
        elif int(val) == 2:
            ret_val = "2"
        elif int(val) == 1:
            ret_val = "1"
        elif int(val) == 0:
            ret_val = "0"
    return ret_attr, ret_val


def display_aux(tree):
    """Returns list of strings, width, height, and horizontal coordinate of the root."""
    # No child. Leaf Node
    if hasattr(tree, 'pred_class'):
        line = f'{tree.pred_class} ' + "({:.2f})".format(tree.prob)
        width = len(line)
        height = 1
        middle = width // 2
        return [line], width, height, middle

    # Two children.
    if hasattr(tree, 'child_ge'):
        left, n, p, x = display_aux(tree.child_ge)
    if hasattr(tree, 'child_lt'):
        right, m, q, y = display_aux(tree.child_lt)
    if hasattr(tree, 'pred_class'):
        s = '%s (%s)' % tree.pred_class % tree.prob
    else:
        attr, val = rollback_data(
            tree.test_attr_name, tree.test_attr_threshold)
        if type(val) == type(str()):
            s = f'{attr}'
        else:
            s = f'{attr} at: ' + "{:.2f}".format(val)
    u = len(s)
    first_line = (x + 1) * ' ' + (n - x - 1) * \
        '_' + s + y * '_' + (m - y) * ' '
    second_line = x * ' ' + '/' + \
        (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
    if p < q:
        left += [n * ' '] * (q - p)
    elif q < p:
        right += [m * ' '] * (p - q)
    zipped_lines = zip(left, right)
    lines = [first_line, second_line] + \
        [a + u * ' ' + b for a, b in zipped_lines]
    return lines, n + m + u, max(p, q) + 2, n + u // 2


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc*len(examples))
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""

    def classify(self, example):
        pass


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            child_miss: DecisionNode or LeafNode representing examples that are missing a
                value for test_attr_name
        """
        self.test_attr_name = test_attr_name
        self.test_attr_threshold = test_attr_threshold
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        """Classify an example based on its test attribute value.

        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold)


class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the whole tree
        """
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        # probability of having the class label
        self.prob = pred_class_count / total_count

    def classify(self, example):
        """Classify an example.

        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count,
                                             self.total_count, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        # build the tree!
        self.root = self.learn_tree(examples)

    def learn_tree(self, examples):
        """Build the decision tree based on entropy and information gain.

        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.

        Returns: a DecisionNode or LeafNode representing the tree
        """
        if self.entropy(examples) == 0:
            max_info = self.major_label(examples)
            return LeafNode(max_info['max_label'], max_info['count'], max_info['total'])
        else:
            split = self.best_split(examples)
            if split['leaf_node']:
                max_info = self.major_label(examples)
                return LeafNode(max_info['max_label'], max_info['count'], max_info['total'])
            print(split['attr_name'], split['threshold'])
            if len(split['miss_list']) < self.min_leaf_count:
                return DecisionNode(split['attr_name'], split['threshold'], self.learn_tree(split['less_than_list']), self.learn_tree(split['ge_list']), self.learn_tree(random.choice([split['ge_list'], split['less_than_list']])))

            max_info = self.major_label(split['ge_list']) if len(split['ge_list']) >= len(
                split['less_than_list']) else self.major_label(split['less_than_list'])
            return DecisionNode(split['attr_name'], split['threshold'], self.learn_tree(split['less_than_list']), self.learn_tree(split['ge_list']), self.learn_tree(random.choice([split['ge_list'], split['less_than_list']])))

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        return self.root.classify(example)

    def best_split(self, list):
        info_gain = 0
        max_info_gain = -math.inf
        parent_entropy = 0
        ge_child_entropy = 0
        less_than_child_entropy = 0
        best_split_dict = {}
        for attr_name in list[0].keys():
            if attr_name == self.id_name or attr_name == self.class_name:
                continue
            attr_val_list = [li[attr_name] for li in list]
            for test_threshold in attr_val_list:
                ge_list = []
                less_than_list = []
                miss_list = []
                parent_entropy = self.entropy(list)
                if test_threshold == None:
                    continue
                for e in list:
                    if e[attr_name] == None:
                        miss_list.append(e)
                    elif e[attr_name] >= test_threshold:
                        ge_list.append(e)
                    else:
                        less_than_list.append(e)
                ge_child_entropy = len(ge_list)/len(list) * \
                    self.entropy(ge_list)
                less_than_child_entropy = (
                    len(less_than_list)/len(list)) * self.entropy(less_than_list)
                miss_child_entropy = (
                    len(miss_list)/len(list)) * self.entropy(miss_list)
                info_gain = parent_entropy - \
                    (ge_child_entropy + less_than_child_entropy)
                if (info_gain > max_info_gain + miss_child_entropy):
                    if len(ge_list) < self.min_leaf_count or len(less_than_list) < self.min_leaf_count:
                        if 'threshold' in best_split_dict.keys():
                            best_split_dict['leaf_node'] = False
                            continue
                        else:
                            best_split_dict['leaf_node'] = True
                            continue
                    max_info_gain = info_gain
                    best_split_dict['ge_list'] = ge_list
                    best_split_dict['less_than_list'] = less_than_list
                    best_split_dict['miss_list'] = miss_list
                    best_split_dict['attr_name'] = attr_name
                    best_split_dict['threshold'] = test_threshold
                    if(max_info_gain == parent_entropy or len(best_split_dict['ge_list']) < self.min_leaf_count or len(best_split_dict['less_than_list']) < self.min_leaf_count):
                        best_split_dict['leaf_node'] = True
                    else:
                        best_split_dict['leaf_node'] = False
        return best_split_dict

    def entropy(self, list):
        entropy = 0
        if(len(list) == 0):
            return 0
        label_list = [li[self.class_name] for li in list]
        yes = label_list.count("Y")
        no = label_list.count("N")
        yes_prob = yes/len(list)
        no_prob = no/len(list)
        entropy = -(yes_prob*self.log_func(yes_prob) +
                    no_prob*self.log_func(no_prob))
        return entropy

    def log_func(self, num):
        if num == 0:
            return 0
        else:
            return math.log(num)

    def major_label(self, list):
        label_list = [li[self.class_name] for li in list]
        yes = label_list.count("Y")
        no = label_list.count("N")
        max_info = max([yes, no])
        if max_info == yes:
            return {'max_label': "Y", "count": yes, "total": len(list)}
        else:
            return {'max_label': "N", "count": no, "total": len(list)}

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)


def confusion4x4(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([v for v in vals.values()])
    abbr = ["".join(w[0] for w in lab.split()) for lab in labels]
    s = ""
    s += " actual ___________________________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [vals.get((labp, laba), 0)/n for laba in labels]
        s += "       |        |        |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | {:5.2f}  | {:5.2f}  | \n".format(
            ab, *row)
        s += "       |________|________|________|________| \n"
    s += "          {:^4s}     {:^4s}     {:^4s}     {:^4s} \n".format(*abbr)
    s += "                     predicted \n"
    return s


#############################################


if __name__ == '__main__':
    path_to_csv = './data/train.csv'
    class_attr_name = 'Loan_Status'
    id_attr_name = 'Loan_ID'
    min_examples = 10  # minimum number of examples for a leaf node

    # read in the data
    examples = clean_data(read_data(path_to_csv))
    train_examples, test_examples = train_test_split(examples, 0.15)
    test_data = clean_data(read_data('./data/test.csv'))

    # learn a tree from the training set
    tree = DecisionTree(train_examples, id_attr_name,
                        class_attr_name, min_examples)

    # test the tree on the test set and see how we did
    correct = 0
    ordering = ['Y', 'N']  # used to count "almost" right
    test_act_pred = {}
    for example in test_examples:
        actual = example[class_attr_name]
        pred, prob = tree.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[id_attr_name] + ':',
                                                                  "'" + pred + "'", prob,
                                                                  "'" + actual + "'",
                                                                  '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        test_act_pred[(actual, pred)] = test_act_pred.get(
            (actual, pred), 0) + 1
    print("\n\n\n\nTEST DATA\n")

    for example in test_data:
        pred, prob = tree.classify(example)
        print("{:30} pred {:15} ({:.2f})".format(example[id_attr_name] + ':',
                                                 "'" + pred + "'", prob,))

    print("\naccuracy: {:.2f}".format(correct/len(test_examples)))
    # print(confusion4x4(['Y', 'N'], test_act_pred))
    if os.path.exists("tree.txt"):
        os.remove("tree.txt")
    f = open("tree.txt", "x")
    f.write("\n\n")
    for line in display_aux(tree.root)[0]:
        f.write(line + "\n")
    f.close()
    # print(tree)  # visualize the tree in sweet, 8-bit text
