import math
import pandas as pd
import numpy as np


class AdaboostDecisionStumps:

    def __init__(self, num_stumps, weighted_gini=False):
        self.num_stumps = num_stumps
        self.classes = []
        self.class_heading = ""
        self.decision_stumps = []
        self.weighted_gini = weighted_gini
        self.weights_initialised = False

    def fit(self, X, y):
        # Clear the decision stumps
        self.decision_stumps = []

        attributes = X.columns
        self.classes = np.unique(y)
        self.class_heading = y.name

        for i in range(self.num_stumps):
            # Array to hold threshold value and gini score [(threshold, gini_score)]
            gini_scores = []

            data = pd.concat([X, y], axis=1)

            if not (self.weighted_gini and self.weights_initialised):
                data["weight"] = 1 / len(data)
                self.weights_initialised = True

            for a in attributes:
                gini_scores.append(self.find_best_threshold(data, a))

            # Find the smallest gini score
            min_gini_score = min(list(zip(*gini_scores))[1])

            # get index of smallest gini score
            index = list(zip(*gini_scores))[1].index(min_gini_score)

            # Attribute that provides the best gini score
            best_attribute = attributes[index]

            # Best threshold for that attribute
            threshold = gini_scores[index][0]

            # Make a decision stump using the best threshold for the best attribute
            decision_stump = DecisionStump(data, best_attribute, threshold, self.classes, self.class_heading)

            data["correctly_classified"] = np.nan

            for j in range(0, len(data)):
                if data[self.class_heading][j] in decision_stump.vote(data[best_attribute][j])[0]:
                    # Correctly classified
                    data.loc[j, "correctly_classified"] = True
                else:
                    # Incorrectly classified
                    data.loc[j, "correctly_classified"] = False

            incorrect_classifications = len(data[data["correctly_classified"] == False])

            error = incorrect_classifications / len(data)

            # Set the weight for the stump depending on the error rate
            if error == 0:
                decision_stump.weight = 1
            elif error == 1:
                decision_stump.weight = -1
            else:
                decision_stump.weight = 0.5 * math.log((1 - error) / error, 10)

            self.decision_stumps.append(decision_stump)

            # Increase weights of incorrectly classified
            # Decrease weights of correctly classified
            data.loc[data["correctly_classified"] == True, ["weight"]] *= (math.e ** -decision_stump.weight)
            data.loc[data["correctly_classified"] == False, ["weight"]] *= (math.e ** decision_stump.weight)

            # Normalise weights
            data["weight"] = data["weight"] / sum(data["weight"])

            if self.weighted_gini:
                X = data.drop(self.class_heading, axis=1)
                y = data[self.class_heading]
            else:
                new_data = data.sample(frac=1, replace=True, weights=data["weight"])
                new_data = new_data.reset_index(drop=True)
                X = new_data.drop(self.class_heading, axis=1)
                y = new_data[self.class_heading]

    def find_best_threshold(self, data, attribute):

        # Sort the data, reset the index
        data = data.sort_values(by=attribute)
        data = data.reset_index(drop=True)

        if self.weighted_gini:
            # Represents the total weight of cases in the left leaf for every possible split
            # Represents the total weight of cases in the right leaf for every possible split when reversed
            total = pd.DataFrame(np.ones(len(data)))
            total *= data["weight"]
            total = total.cumsum()
        else:
            # Represents the total number of cases in the left leaf for every possible split
            # Represents the total number of cases in the right leaf for every possible split when reversed
            total = pd.DataFrame(np.ones((len(data))))
            total = total.cumsum()

        # Each row represents number of each class in left leaf for every possible split
        left = pd.DataFrame(np.zeros((len(data), len(self.classes))), columns=self.classes)
        # Each row represents number of each class in right leaf for every possible split
        right = pd.DataFrame(np.zeros((len(data), len(self.classes))), columns=self.classes)

        if self.weighted_gini:
            # For each class, put case weight in cell if case belongs to class
            # i.e
            # 1st case is of class 'X' and has weight 0.2
            # 2nd case is of class 'X' and has weight 0.15
            # 3rd case is of class 'Z' and has weight 0.2
            #       X       Y       Z
            #       0.2     0       0
            #       0.15    0       0
            #       0       0       0.2
            for c in self.classes:
                left.loc[data[self.class_heading] == c, [c]] = data["weight"]
        else:
            # For each class, put 1 in cell if case belongs to class
            # i.e
            # 1st case is of class 'X'
            # 2nd case is of class 'X'
            # 3rd case is of class 'Z'
            #       X       Y       Z
            #       1       0       0
            #       1       0       0
            #       0       0       1
            for c in self.classes:
                left.loc[data[self.class_heading] == c, [c]] = 1

        # Cumulative sum each column
        # For each column, every row is the sum of itself + all previous rows
        # i.e
        #       X       Y       Z               X       Y       Z
        #       1       0       0               1       0       0
        #       1       0       0       ->      2       0       0
        #       0       0       1               2       0       1
        left = left.cumsum()

        # Fill right with the values of the last row of left
        # i.e
        #       X       Y       Z
        #       2       0       1
        #       2       0       1
        #       2       0       1
        right += left.iloc[len(left) - 1]

        # Subtract left from right to get proper values for right
        # i.e
        #       X       Y       Z
        #       1       0       1
        #       0       0       1
        #       0       0       0
        right = right - left

        # Drop the last rows as they represent all the data in the left leaf and none in the right
        total = total.drop(len(total) - 1, axis=0)
        left = left.drop(len(total), axis=0)
        right = right.drop(len(total), axis=0)

        # Calculate the gini index for every possible split
        left["gini"] = 1 - np.sum(np.divide(left, total) ** 2, axis=1)
        right["gini"] = 1 - np.sum(np.divide(right, total.iloc[::-1]) ** 2, axis=1)

        total_gini = \
            np.divide(total, len(data)).mul(left["gini"], axis=0) + \
            np.divide(total.iloc[::-1], len(data)).reset_index(drop=True).mul(right["gini"], axis=0)

        # Index of smallest gini
        index = total_gini.idxmin()[0]

        threshold = (data[attribute][index] + data[attribute][index+1]) / 2

        return (threshold, total_gini.iloc[index][0])

    def classify(self, case):
        # Array to hold votes [([classes], weight)]
        votes = []
        for s in self.decision_stumps:
            votes.append(s.vote(case[s.attribute]))

        # Array to hold vote values for each class [class1: x, class2: y, ...]
        class_votes = []
        for c in self.classes:
            vote_count = 0
            for v in votes:
                if c in v[0]:
                    # If the class is voted for by a stump, ass the stumps weight to the vote count
                    vote_count += v[1]
            class_votes.append(vote_count)
        # Return class with the highest vote value
        return self.classes[class_votes.index(max(class_votes))]

    def score(self, X, y, output_to_file=False):
        score = 0
        results = []
        output = open("results.txt", "w")

        for i in range(len(X)):
            classified_as = self.classify(X.iloc[i])
            actual_class = y.iloc[i]
            if output_to_file:
                results.append((classified_as, actual_class))
            if classified_as == actual_class:
                score += 1

        score /= len(X)

        if output_to_file:
            output.write("Number of cases: '{0}'\n"
                         "weighted_gini='{1}'\n"
                         "Classified as\tActual class\n".format(len(X), self.weighted_gini))
            for r in results:
                output.write("'{0}'\t\t'{1}'\n".format(r[0], r[1]))

        return score


class DecisionStump:

    def __init__(self, data, attribute, threshold, classes, class_heading):
        self.attribute = attribute
        self.threshold = threshold
        self.weight = 0  # Weight is assigned after stump is created
        self.left_classes = []
        self.right_classes = []

        left = data[data[self.attribute] < self.threshold]
        right = data[data[self.attribute] >= self.threshold]

        for c in classes:
            left_count = len(left[left[class_heading] == c]) / len(left)
            right_count = len(right[right[class_heading] == c]) / len(right)

            if left_count > right_count:
                self.left_classes.append(c)
            else:
                self.right_classes.append(c)

    def vote(self, value):
        if value < self.threshold:
            return self.left_classes, self.weight
        else:
            return self.right_classes, self.weight

    def print(self):
        print("\nAttribute: ", self.attribute,
              "\nThreshold: ", self.threshold,
              "\nWeight", self.weight,
              "\nLeft classes: ", self.left_classes,
              "\nRight classes: ", self.right_classes)
