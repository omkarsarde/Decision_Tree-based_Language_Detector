"""
English or Dutch predictor using Decision Tree and Adaboost
This code is built on the skeleton described in data science course (PGP AI) from edureka.co
"""
import math
import pickle
import sys
import time


class EnglishDutchPredictor:

    def gather_data(self, file):
        """
        Gathers data from the train.txt file for training
        :param file:input training file
        :return:ground truth and statements
        """
        file_details = open(file, encoding="utf-8-sig")
        all_details = ''
        for file_lines in file_details:
            all_details += file_lines
        statements = all_details.split('|')
        all_data_stripped_space = all_details.split()

        for index in range(len(statements)):
            if index < 1:
                continue
            statements[index] = statements[index][:-4]
        statements = statements[1:]
        ground_truth = []
        pointer = 0
        for values in all_data_stripped_space:
            if values.startswith('nl|') or values.startswith('en|'):
                ground_truth.insert(pointer, values[0:2])
                pointer = pointer + 1
        return statements, ground_truth

    def decision_tree_predict(self, hypothesis, file):
        """
        Prediction using decision tree
        :param hypothesis: model which has been trained
        :param file: test data set
        :return:
        """
        object = pickle.load(open(hypothesis, 'rb'))
        file_open = open(file)
        sentence_list = []
        counter = 0
        sentence = ''
        for line in file_open:
            words = line.split()

            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
                    sentence += word
                    sentence_list.append(sentence)
                    sentence = ''
                    counter = 0
        statements, ground_truth = self.gather_data(file)
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = ([] for i in range(17))
        features = []
        for i in sentence_list:
            a1.append(self.en_at1(i))
            a2.append(self.en_at2(i))
            a3.append(self.en_at3(i))
            a4.append(self.en_at4(i))
            a5.append(self.en_at5(i))
            a6.append(self.en_at6(i))
            a7.append(self.en_at7(i))
            a8.append(self.en_at8(i))
            a9.append(self.en_at9(i))
            a10.append(self.en_at10(i))
            a11.append(self.nl_at1(i))
            a12.append(self.nl_at2(i))
            a13.append(self.nl_at3(i))
            a14.append(self.nl_at4(i))
            a15.append(self.nl_at5(i))
            a16.append(self.nl_at6(i))
            a17.append(self.nl_at7(i))
        features = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]
        counter = 0
        for i in sentence_list:
            object_temp = object
            while type(object_temp.value) != str:
                value = features[object_temp.value][counter]
                if value is True:
                    object_temp = object_temp.left
                else:
                    object_temp = object_temp.right
            print(object_temp.value)
            counter = counter + 1

    def decision_tree_train(self, example_file, hypothesis_file):
        """
        training of decision tree
        :param example_file:training dataset
        :param hypothesis_file: model written on this file
        :return:None
        """
        start_time = time.time()
        statements, ground_truth = self.gather_data(example_file)
        print(str(len(ground_truth)) + " sentences for training.")
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = ([] for i in range(17))
        attributes = []
        for i in statements:
            a1.append(self.en_at1(i))
            a2.append(self.en_at2(i))
            a3.append(self.en_at3(i))
            a4.append(self.en_at4(i))
            a5.append(self.en_at5(i))
            a6.append(self.en_at6(i))
            a7.append(self.en_at7(i))
            a8.append(self.en_at8(i))
            a9.append(self.en_at9(i))
            a10.append(self.en_at10(i))
            a11.append(self.nl_at1(i))
            a12.append(self.nl_at2(i))
            a13.append(self.nl_at3(i))
            a14.append(self.nl_at4(i))
            a15.append(self.nl_at5(i))
            a16.append(self.nl_at6(i))
            a17.append(self.nl_at7(i))
        attributes = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]
        train_sentence_num = []
        for i in range(len(ground_truth)):
            train_sentence_num.append(i)
        seen = []
        root = TreeNode(attributes, None, ground_truth, train_sentence_num, 0, None, None)
        self.create_tree(root, attributes, seen, ground_truth, train_sentence_num, 0, None)
        file = open(hypothesis_file, 'wb')
        pickle.dump(root, file)
        print("Training completed")
        print("Time Taken: " + str(time.time() - start_time) + " seconds")

    def plurality(self, attribute, total_sentence_num):
        """
        return if types of values are different
        :param attribute: features
        :param total_sentence_num: number of training sentences
        :return: Check if data set is not SINGULAR
        """
        value = attribute[total_sentence_num[0]]
        for i in total_sentence_num:
            if value != attribute[i]:
                return 10
        return 0

    def entropy(self, value):
        """
        Entropy function
        :param value:Input value
        :return:Entropy
        """
        if value == 1:
            return 0
        return (-1) * (value * math.log2(value) + (1 - value) * math.log2(1 - value))

    def singularity(self, values):
        """
        Check if data set has all same max gain at 0
        :param values:values
        :return:False if max gain is 0
        """
        if max(values) == 0:
            return False

    def create_tree(self, root, attributes, seen, ground_truth, total_ground_truth, depth, prediction):
        """
        Create a recursive decision tree
        :param root: current node
        :param attributes:Features
        :param seen: nodes visited till this level
        :param ground_truth:Final ground_truth
        :param total_ground_truth:number of train cases till this level
        :param depth:Level in consideration
        :param prediction:Prediction made
        :return:None
        """
        if depth == len(attributes) - 1:
            count_en = 0
            count_nl = 0
            for index in total_ground_truth:
                if ground_truth[index] is 'en':
                    count_en = count_en + 1
                elif ground_truth[index] is 'nl':
                    count_nl = count_nl + 1
            if count_en > count_nl:
                root.value = 'en'
            else:
                root.value = 'nl'
        elif len(total_ground_truth) == 0:
            root.value = prediction
        elif self.plurality(ground_truth, total_ground_truth) == 0:
            root.value = ground_truth[total_ground_truth[0]]
        elif len(attributes) == len(seen):
            count_en = 0
            count_nl = 0
            for index in total_ground_truth:
                if ground_truth[index] is 'en':
                    count_en = count_en + 1
                elif ground_truth[index] is 'nl':
                    count_nl = count_nl + 1
            if count_en > count_nl:
                root.value = 'en'
            else:
                root.value = 'nl'
        else:
            gain = []
            ground_truth_en = 0
            ground_truth_nl = 0
            for index in total_ground_truth:
                if ground_truth[index] == 'en':
                    ground_truth_en = ground_truth_en + 1
                else:
                    ground_truth_nl = ground_truth_nl + 1
            for i1 in range(len(attributes)):
                if i1 in seen:
                    gain.append(0)
                    continue
                else:
                    count_true_en, count_true_nl, count_false_en, count_false_nl = (0 for i in range(4))
                    for i2 in total_ground_truth:
                        if attributes[i1][i2] is True and ground_truth[i2] == 'en':
                            count_true_en = count_true_en + 1
                        elif attributes[i1][i2] is True and ground_truth[i2] == 'nl':
                            count_true_nl = count_true_nl + 1
                        elif attributes[i1][i2] is False and ground_truth[i2] == 'en':
                            count_false_en = count_false_en + 1
                        elif attributes[i1][i2] is False and ground_truth[i2] == 'nl':
                            count_false_nl = count_false_nl + 1
                    if (count_true_nl + count_true_en == 0) or (count_false_en + count_false_nl == 0):
                        current_gain = 0
                        gain.append(current_gain)
                        continue
                    if count_true_en == 0:
                        false_total = count_false_nl + count_false_en
                        total_set = ground_truth_nl + ground_truth_nl
                        fraction = count_false_en / false_total
                        rem_true_value = 0
                        rem_false_value = (false_total / total_set) * self.entropy(fraction)
                    elif count_false_en == 0:
                        rem_false_value = 0
                        true_total = count_true_en + count_true_nl
                        total_set = ground_truth_nl + ground_truth_en
                        fraction = count_true_en / true_total
                        rem_true_value = (true_total / total_set) * self.entropy(fraction)
                    else:
                        true_total = count_true_en + count_true_nl
                        false_total = count_false_en + count_false_nl
                        total_set = ground_truth_en + ground_truth_nl
                        fraction_true = count_true_en / true_total
                        fraction_false = count_false_en / false_total
                        rem_true_value = (true_total / total_set) * self.entropy(fraction_true)
                        rem_false_value = (false_total / total_set) * self.entropy(fraction_false)

                    re_total = ground_truth_en + ground_truth_nl
                    fraction_en = ground_truth_en / re_total
                    total_rem = rem_false_value + rem_true_value
                    current_gain = self.entropy(fraction_en) - total_rem
                    gain.append(current_gain)
            continue_var = self.singularity(gain)
            if continue_var is False:
                root.value = prediction
                return
            max_gain_attribute = gain.index(max(gain))
            seen.append(max_gain_attribute)
            index_True = []
            index_False = []
            for index in total_ground_truth:
                if attributes[max_gain_attribute][index] is True:
                    index_True.append(index)
                else:
                    index_False.append(index)
            if ground_truth_en > ground_truth_nl:
                prediction_at_this_stage = 'en'
            else:
                prediction_at_this_stage = 'nl'
            bool_false = False
            bool_true = True
            root.value = max_gain_attribute
            left_child = TreeNode(attributes, None, ground_truth, index_True, depth + 1,
                                  prediction_at_this_stage, bool_true)
            right_child = TreeNode(attributes, None, ground_truth, index_False, depth + 1,
                                   prediction_at_this_stage, bool_false)
            root.left = left_child
            root.right = right_child
            self.create_tree(left_child, attributes, seen, ground_truth, index_True, depth + 1,
                             prediction_at_this_stage)
            self.create_tree(right_child, attributes, seen, ground_truth, index_False, depth + 1,
                             prediction_at_this_stage)

            del seen[-1]

    def adaboost_training(self, example_file, hypothesis_file):
        """
        Adaboost training along with data collection
        :param example_file:Training file for training
        :param hypothesis_file: model file to write hypothesis on
        :return:None
        """
        start_time = time.time()
        statements, ground_truth = self.gather_data(example_file)
        weights = [1 / len(statements)] * len(statements)
        number_of_decision_stumps = 50
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = ([] for i in range(17))
        for i in statements:
            a1.append(self.en_at1(i))
            a2.append(self.en_at2(i))
            a3.append(self.en_at3(i))
            a4.append(self.en_at4(i))
            a5.append(self.en_at5(i))
            a6.append(self.en_at6(i))
            a7.append(self.en_at7(i))
            a8.append(self.en_at8(i))
            a9.append(self.en_at9(i))
            a10.append(self.en_at10(i))
            a11.append(self.nl_at1(i))
            a12.append(self.nl_at2(i))
            a13.append(self.nl_at3(i))
            a14.append(self.nl_at4(i))
            a15.append(self.nl_at5(i))
            a16.append(self.nl_at6(i))
            a17.append(self.nl_at7(i))
        attributes = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]
        training_sentence_num = []
        stump_values = []
        hypothesis_weights = [1] * number_of_decision_stumps
        for i in range(len(ground_truth)):
            training_sentence_num.append(i)
        for hypothesis in range(0, number_of_decision_stumps):
            root = TreeNode(attributes, None, ground_truth, training_sentence_num, 0, None, None)
            stump = self.return_stump(0, root, attributes, ground_truth, training_sentence_num, weights)
            error, correct, incorrect = (0 for i in range(3))
            for index in range(len(statements)):
                if self.prediction(stump, statements[index], attributes, index) != ground_truth[index]:
                    error = error + weights[index]
                    incorrect = incorrect + 1
            for index in range(len(statements)):
                if self.prediction(stump, statements[index], attributes, index) == ground_truth[index]:
                    weights[index] = weights[index] * error / (1 - error)
                    correct = correct + 1
            total = 0
            for weight in weights:
                total += weight
            for index in range(len(weights)):
                weights[index] = weights[index] / total
            hypothesis_weights[hypothesis] = math.log(((1 - error) / (error)), 2)
            stump_values.append(stump)
        filehandler = open(hypothesis_file, 'wb')
        pickle.dump((stump_values, hypothesis_weights), filehandler)
        print("Training completed using ADABOOST")
        print("Time Taken: " + str(time.time() - start_time) + " seconds")

    def prediction(self, stump, statement, attributes, index):
        """
        For predicting the stump and the result it will give
        :param stump:Input decision stump
        :param statement:Input statement
        :param attributes: features
        :param index: current level
        :return:Return prediction from the stump
        """
        attribute_value = stump.value
        if attributes[attribute_value][index] is True:
            return stump.left.value
        else:
            return stump.right.value

    def return_stump(self, depth, root, attributes, ground_truth, training_sentence_num, weights):
        """
        Function returns a decision stump
        :param depth: current level
        :param root:
        :param attributes:features
        :param ground_truth: ground truth table
        :param training_sentence_num: number of train cases trained before this level
        :param weights: weight
        :return:
        """
        gain = []
        ground_truth_en = 0
        ground_truth_nl = 0
        for index in training_sentence_num:
            if ground_truth[index] == 'en':
                ground_truth_en = ground_truth_en + 1 * weights[index]
            else:
                ground_truth_nl = ground_truth_nl + 1 * weights[index]

        for i1 in range(len(attributes)):
            count_true_en, count_true_nl, count_false_en, count_false_nl = (0 for i in range(4))
            for index in training_sentence_num:
                if attributes[i1][index] is True and ground_truth[index] == 'en':
                    count_true_en = count_true_en + 1 * weights[index]
                elif attributes[i1][index] is True and ground_truth[index] == 'nl':
                    count_true_nl = count_true_nl + 1 * weights[index]
                elif attributes[i1][index] is False and ground_truth[index] == 'en':
                    count_false_en = count_false_en + 1 * weights[index]
                elif attributes[i1][index] is False and ground_truth[index] == 'nl':
                    count_false_nl = count_false_nl + 1 * weights[index]
            if count_true_en == 0:
                false_total = count_false_nl + count_false_en
                total_set = ground_truth_nl + ground_truth_nl
                fraction = count_false_en / false_total
                rem_true_value = 0
                rem_false_value = (false_total / total_set) * self.entropy(fraction)
            elif count_false_en == 0:
                rem_false_value = 0
                true_total = count_true_en + count_true_nl
                total_set = ground_truth_nl + ground_truth_en
                fraction = count_true_en / true_total
                rem_true_value = (true_total / total_set) * self.entropy(fraction)
            else:
                true_total = count_true_en + count_true_nl
                false_total = count_false_en + count_false_nl
                total_set = ground_truth_en + ground_truth_nl
                fraction_true = count_true_en / true_total
                fraction_false = count_false_en / false_total
                rem_true_value = (true_total / total_set) * self.entropy(fraction_true)
                rem_false_value = (false_total / total_set) * self.entropy(fraction_false)
            re_total = ground_truth_en + ground_truth_nl
            fraction_en = ground_truth_en / re_total
            total_rem = rem_false_value + rem_true_value
            current_gain = self.entropy(fraction_en) - total_rem
            gain.append(current_gain)
        max_gain_attribute = gain.index(max(gain))
        root.value = max_gain_attribute
        count_max_true_en = 0
        count_max_true_nl = 0
        count_max_false_en = 0
        count_max_false_nl = 0
        for index in range(len(attributes[max_gain_attribute])):
            if attributes[max_gain_attribute][index] is True:
                if ground_truth[index] == 'en':
                    count_max_true_en = count_max_true_en + 1 * weights[index]
                else:
                    count_max_true_nl = count_max_true_nl + 1 * weights[index]
            else:
                if ground_truth[index] == 'en':
                    count_max_false_en = count_max_false_en + 1 * weights[index]
                else:
                    count_max_false_nl = count_max_false_nl + 1 * weights[index]
        left_child = TreeNode(attributes, None, ground_truth, None, depth + 1,
                              None, None)
        right_child = TreeNode(attributes, None, ground_truth, None, depth + 1,
                               None, None)
        if count_max_true_en > count_max_true_nl:
            left_child.value = 'en'
        else:
            left_child.value = 'nl'
        if count_max_false_en > count_max_false_nl:
            right_child.value = 'en'
        else:
            right_child.value = 'nl'

        root.left = left_child
        root.right = right_child
        return root

    def adaboost_predict(self, hypothesis_file, input_test_file):
        """
        Making the prediction using the saved adaboost model
        :param hypothesis_file:File containing the adaboost model
        :param input_test_file:Test file to be tested
        :return:None
        """
        object = pickle.load(open(hypothesis_file, 'rb'))
        file_open = open(input_test_file)
        sentence_list = []
        counter = 0
        sentence = ''
        for line in file_open:
            words = line.split()
            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
                    sentence += word
                    sentence_list.append(sentence)
                    sentence = ''
                    counter = 0
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = ([] for i in range(17))
        for i in sentence_list:
            a1.append(self.en_at1(i))
            a2.append(self.en_at2(i))
            a3.append(self.en_at3(i))
            a4.append(self.en_at4(i))
            a5.append(self.en_at5(i))
            a6.append(self.en_at6(i))
            a7.append(self.en_at7(i))
            a8.append(self.en_at8(i))
            a9.append(self.en_at9(i))
            a10.append(self.en_at10(i))
            a11.append(self.nl_at1(i))
            a12.append(self.nl_at2(i))
            a13.append(self.nl_at3(i))
            a14.append(self.nl_at4(i))
            a15.append(self.nl_at5(i))
            a16.append(self.nl_at6(i))
            a17.append(self.nl_at7(i))
        features = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]

        statement_pointer = 0
        hypothesis_weights = object[1]
        hypothesis_list = object[0]
        for sentence in sentence_list:
            total_summation = 0
            for index in range(len(object[0])):
                total_summation += self.adaboost_predict_helper(hypothesis_list[index], sentence, features,
                                                                statement_pointer) * hypothesis_weights[index]

            if total_summation > 0:
                print('en')
            else:
                print('nl')
            statement_pointer += 1

    def adaboost_predict_helper(self, stump, sentence, features, index):
        """
        Returns prediction based on the input hypothesis(stump) in consideration
        :param stump:Input hypothesis
        :param sentence:Input sentence
        :param features: features
        :param index: current index
        :return:
        """
        attribute_value = stump.value
        if features[attribute_value][index] is True:
            if stump.left.value == 'en':
                return 1
            else:
                return -1
        else:
            if stump.right.value == 'en':
                return 1
            else:
                return -1

    # features
    def en_at1(self, in_sentence):
        # articles a an the
        determiners = "a an the".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at2(self, in_sentence):
        # determiners
        determiners = "this that these those my your his its our their".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at3(self, in_sentence):
        # quantifiers and interrogative
        determiners = "all every most many much some few little any no double twice triple thrice".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at4(self, in_sentence):
        # prepositions
        determiners = "of in to for with on at from by about as into like through after over between out" \
                      " against during without before under around among".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at5(self, in_sentence):
        # pronouns
        determiners = "i me you he him her she this that these those mine yours his hers its who what which whom whose" \
                      " myself yourself himself herself other another much nobody anybody few such".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at6(self, in_sentence):
        # common verbs
        determiners = "be have do say get make go know take see come think look want give use find tell" \
                      " ask work seem feel try leave all".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at7(self, in_sentence):
        # common conjunctions
        determiners = "and that but or as if when than because while where after so though since until whether" \
                      " before although nor like once unless now except".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at8(self, in_sentence):
        # common tenses
        determiners = "have had will would be been am ing".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in sentence:
            if i in determiners:
                return True
        return False

    def en_at9(self, in_sentence):
        # common adverbs
        determiners = "up so out just now how then more also here well only very even back there down still" \
                      " in as too when never really most".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def en_at10(self, in_sentence):
        # common adjectives
        determiners = "good new first last long great little own other old right big high different small" \
                      " large next early young important few public bad same able".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at1(self, in_sentence):
        # articles de het een
        determiners = "de het een".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at2(self, in_sentence):
        # common pronouns
        determiners = "ik jij je hij wij jullie jezelf zich zichzelf het hem ze dit deze" \
                      " dat die wat iets niets alles".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at3(self, in_sentence):
        # common adverbs
        determiners = "er hier daar waar ergens nergens overal zijn waren".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at4(self, in_sentence):
        # common conjunctions
        determiners = "als alsof behalve dat hoewel nu nudat omdat na nadat sinds tenzij terwijl toen" \
                      " tot totdat wanneer voor voordat zoals zodat zolang zonder zover".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at5(self, in_sentence):
        # prepositions
        determiners = "boven tegen onder rond als op voor zjin hebben gaan ga komen kom willen zullen zitten" \
                      " nemen blijven lopen kunnen moeten mogen staan weten kijken kijkt maken maakt doen werken" \
                      " werkt zegt horen luisteren zein kijken weten leren slapen lezen schrijven spreken praten" \
                      " zeggen denken werken wachten naar be ik het voor niet met hij zijn ze wij ze er hun zo hem" \
                      " weten jouw dan ook onze deze ons meest".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False

    def nl_at6(self, in_sentence):
        # er erkt en is (dunno why they occur the most)
        determiners = "er erkt en de is den te ten".split()
        sentence = in_sentence.lower()
        # words = sentence.split()
        for i in sentence:
            if i in determiners:
                return True
        return False

    def nl_at7(self, in_sentence):
        # van (dunno why they occur the most)
        determiners = "van".split()
        sentence = in_sentence.lower()
        words = sentence.split()
        for i in words:
            if i in determiners:
                return True
        return False


class TreeNode:
    def __init__(self, attributes, seen, ground_truth, training_sentence_num, depth, prediction, boolean):
        self.attributes = attributes
        self.seen = seen
        self.ground_truth = ground_truth
        self.training_sentence_num = training_sentence_num
        self.depth = depth
        self.prediction = prediction
        self.bool = boolean
        self.value = None
        self.left = None
        self.right = None


def main():
    """
    Main Function
    :return: None
    """
    obj = EnglishDutchPredictor()
    # Check if right command line arguments given
    try:
        if sys.argv[1] == 'train':
            if sys.argv[4] == 'dt':
                obj.decision_tree_train(sys.argv[2], sys.argv[3])
            else:
                obj.adaboost_training(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == 'predict':
            if sys.argv[4] == 'dt':
                obj.decision_tree_predict(sys.argv[2], sys.argv[3])
            else:
                obj.adaboost_predict(sys.argv[2], sys.argv[3])
    except:
        print('Syntax :train <examples> <hypothesisOut> <learning-type>')
        print('or')
        print('Syntax :predict <hypothesis> <file> <testing-type(dt or ada)>')


if __name__ == "__main__":
    main()
