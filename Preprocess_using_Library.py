import re
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""
Testing validity of features using prebuilt library
"""
class PreProcessing:

    def __init__(self):
        self.train_sentences = []
        self.test_sentences = []
        self.final = []

    def generate_train_sentences(self, train_file):
        with open(train_file, encoding="utf-8-sig") as f:
            counter = 0
            for line in f:
                if "nl|" in line:
                    nl_line = line.replace("nl|", "")
                    new_line = re.sub('[^A-Za-z0-9]+', ' ', nl_line)
                    # self.nl_sentences.append(new_line)
                    self.final.append("nl")
                    self.train_sentences.append(new_line)
                else:
                    en_line = line.replace("en|", "")
                    new_line = re.sub('[^A-Za-z0-9]+', ' ', en_line)
                    self.final.append("en")
                    self.train_sentences.append(new_line)
                    # self.en_sentences.append(new_line)
        self.generate_attributes()

    def generate_test_sentences(self, test_file):
        with open(test_file, encoding="utf-8-sig") as f:
            for line in f:
                new_line = re.sub('[^A-Za-z0-9]+', ' ', line)
                self.test_sentences.append(new_line)
        self.generate_test_attributes()

    def generate_attributes(self):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17 = ([] for i in range(17))
        final = self.final
        for i in self.train_sentences:
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
        dict = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'a8': a8, 'a9': a9, 'a10': a10,
                'a11': a11, 'a12': a12, 'a13': a13, 'a14': a14, 'a15': a15, 'a16': a16, 'a17': a17, 'lang': final}
        df = pd.DataFrame(dict)
        df.to_csv("TrainTable.csv")

    def generate_test_attributes(self):
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17 = ([] for i in range(17))
        for i in self.test_sentences:
            t1.append(self.en_at1(i))
            t2.append(self.en_at2(i))
            t3.append(self.en_at3(i))
            t4.append(self.en_at4(i))
            t5.append(self.en_at5(i))
            t6.append(self.en_at6(i))
            t7.append(self.en_at7(i))
            t8.append(self.en_at8(i))
            t9.append(self.en_at9(i))
            t10.append(self.en_at10(i))
            t11.append(self.nl_at1(i))
            t12.append(self.nl_at2(i))
            t13.append(self.nl_at3(i))
            t14.append(self.nl_at4(i))
            t15.append(self.nl_at5(i))
            t16.append(self.nl_at6(i))
            t17.append(self.nl_at7(i))
        dict = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t9": t9, "t10": t10,
                "t11": t11, "t12": t12, "t13": t13, "t14": t14, "t15": t15, "t16": t16, "t17": t17}
        df = pd.DataFrame(dict, )
        df.to_csv("TestTable.csv")

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


def main():
    test = PreProcessing("trainingData.txt", "testData.txt")
    test.generate_sentences()
    frame = pd.read_csv("TrainTable.csv")
    test.generate_test_sentences()
    test_frame = pd.read_csv("TestTable.csv")
    # sns.pairplot(frame,hue="lang")
    X = frame.drop('lang', axis=1)
    y = frame['lang']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    dtree = DecisionTreeClassifier(criterion="entropy")
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    # dp = dtree.predict(test_frame)
    print(sklearn.metrics.confusion_matrix(y_test, predictions))
    print("\n for DT")
    print(sklearn.metrics.classification_report(y_test, predictions))
    test = dtree.predict(test_frame)
    # print(test)
    # print("Decision Tree ?")
    # print(dp)
    # RandomForest
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, rfc_pred))
    print("\n for RandomForest")
    print(sklearn.metrics.classification_report(y_test, rfc_pred))
    # out = rfc.predict(test_frame)
    # print("random forest ?")
    # print(out)


if __name__ == "__main__":
    main()
