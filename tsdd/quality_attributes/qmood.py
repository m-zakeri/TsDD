"""
QMOOD Design Metrics and
QMOOD quality attributes

## Reference
[1] J. Bansiya and C. G. Davis, “A hierarchical model for object-oriented design quality assessment,”
IEEE Trans. Softw. Eng., vol. 28, no. 1, pp. 4–17, 2002.

"""

__version__ = '0.1.1'
__author__ = 'Morteza Zakeri'

import os
import understand as und


class QMOODMetric:
    def __init__(self, udb_path):
        self.db = und.open(udb_path)
        self.metrics = self.db.metric(self.db.metrics())

    @property
    def DSC(self):
        """
        DSC - Design Size in Classes
        :return: Total number of classes in the design.
        """
        return self.metrics.get('CountDeclClass', 0)

    @property
    def NOH(self):
        """
        NOH - Number Of Hierarchies
        count(MaxInheritanceTree(class)) = 0
        :return: Total number of 'root' classes in the design.
        """
        count = 0
        for ent in sorted(self.db.ents(kindstring='class'), key=lambda ent: ent.name()):
            if ent.kindname() == "Unknown Class":
                continue
            mit = ent.metric(['MaxInheritanceTree'])['MaxInheritanceTree']
            if mit == 1:
                count += 1
        return count

    @property
    def ANA(self):
        """
        ANA - Average Number of Ancestors
        :return: Average number of classes in the inheritance tree for each class
        """
        MITs = []
        for ent in sorted(self.db.ents(kindstring='class'), key=lambda ent: ent.name()):
            if ent.kindname() == "Unknown Class":
                continue
            mit = ent.metric(['MaxInheritanceTree'])['MaxInheritanceTree']
            MITs.append(mit)
        return sum(MITs) / len(MITs)

    @property
    def MOA(self):
        """
        MOA - Measure of Aggregation
        :return: Count of number of attributes whose type is user defined class(es).
        """
        counter = 0
        user_defined_classes = []
        for ent in sorted(self.db.ents(kindstring="class"), key=lambda ent: ent.name()):
            if ent.kindname() == "Unknown Class":
                continue
            user_defined_classes.append(ent.simplename())
        for ent in sorted(self.db.ents(kindstring="variable"), key=lambda ent: ent.name()):
            if ent.type() in user_defined_classes:
                counter += 1
        return counter

    @property
    def DAM(self):
        """
        DAM - The Average of Direct Access Metric for all classes
        :return: AVG(All Class's DAM).
        """
        return self.get_class_average(self.ClassLevelDAM)

    @property
    def CAMC(self):
        """
        CAMC - Cohesion Among Methods of class
        :return: AVG(All Class's CAMC)
        """
        return self.get_class_average(self.ClassLevelCAMC)

    @property
    def CIS(self):
        """
        CIS - Class Interface Size
        :return: AVG(All class's CIS)
        """
        return self.get_class_average(self.ClassLevelCIS)

    @property
    def NOM(self):
        """
        NOM - Number of Methods
        :return: AVG(All class's NOM)
        """
        return self.get_class_average(self.ClassLevelNOM)

    @property
    def DCC(self):
        """
        DCC - Direct Class Coupling
        :return: AVG(All class's DCC)
        """
        return self.get_class_average(self.ClassLevelDCC)

    @property
    def MFA(self):
        """
        MFA - Measure of Functional Abstraction
        :return: AVG(All class's MFA)
        """
        return self.get_class_average(self.ClassLevelMFA)

    @property
    def NOP(self):
        """
        NOP - Number of Polymorphic Methods
        :return: AVG(All class's NOP)
        """
        return self.get_class_average(self.ClassLevelNOP)

    def ClassLevelDAM(self, class_longname):
        """
        DAM - Class Level Direct Access Metric
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Ratio of the number of private and protected attributes to the total number of attributes in a class.
        """
        public_variables = 0
        protected_variables = 0
        private_variables = 0

        class_entity = self.get_class_entity(class_longname)
        for ref in class_entity.refs(refkindstring="define"):
            define = ref.ent()
            kind_name = define.kindname()
            if kind_name == "Public Variable":
                public_variables += 1
            elif kind_name == "Private Variable":
                private_variables += 1
            elif kind_name == "Protected Variable":
                protected_variables += 1

        try:
            ratio = (private_variables + protected_variables) / (
                    private_variables + protected_variables + public_variables)
        except ZeroDivisionError:
            ratio = 0.0
        return ratio

    def ClassLevelCAMC(self, class_longname):
        """
        CAMC - Class Level Cohesion Among Methods of class
        Measures of how related methods are in a class in terms of used parameters.
        It can also be computed by: 1 - LackOfCohesionOfMethods()
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return:  A float number between 0 and 1.
        """
        class_entity = self.get_class_entity(class_longname)
        percentage = class_entity.metric(['PercentLackOfCohesion']).get('PercentLackOfCohesion', 0)
        if percentage is None:
            percentage = 0
        return 1.0 - percentage / 100

    def ClassLevelCIS(self, class_longname):
        """
        CIS - Class Level Class Interface Size
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Number of public methods in class
        """
        class_entity = self.get_class_entity(class_longname)
        value = class_entity.metric(['CountDeclMethodPublic']).get('CountDeclMethodPublic', 0)
        if value is None:
            value = 0
        return value

    def ClassLevelNOM(self, class_longname):
        """
        NOM - Class Level Number of Methods
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Number of methods declared in a class.
        """
        class_entity = self.get_class_entity(class_longname)
        if class_entity:
            return class_entity.metric(['CountDeclMethod']).get('CountDeclMethod', 0)
        return 0

    def ClassLevelDCC(self, class_longname):
        """
        DCC - Class Level Direct Class Coupling
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Number of other classes a class relates to, either through a shared attribute or a parameter in a method.
        """
        counter = 0
        class_entity = self.get_class_entity(class_longname)
        for ref in class_entity.refs():
            if ref.kindname() == "Couple":
                if ref.isforward():
                    if "class" in ref.ent().kindname().lower():
                        counter += 1
        return counter

    def ClassLevelMFA(self, class_longname):
        """
        MFA - Class Level Measure of Functional Abstraction
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Ratio of the number of inherited methods per the total number of methods within a class.
        """
        class_entity = self.get_class_entity(class_longname)
        metrics = class_entity.metric(['CountDeclMethod', 'CountDeclMethodAll'])
        local_methods = metrics.get('CountDeclMethod')
        all_methods = metrics.get('CountDeclMethodAll')
        if all_methods == 0:
            all_methods = 1
        return (all_methods - local_methods) / all_methods

    def ClassLevelNOP(self, class_longname):
        """
        NOP - Class Level Number of Polymorphic Methods
        Any method that can be used by a class and its descendants.
        :param class_longname: The longname of a class. For examole: package_name.ClassName
        :return: Counts of the number of methods in a class excluding private, static and final ones.
        """
        class_entity = self.get_class_entity(class_longname)
        instance_methods = class_entity.metric(['CountDeclInstanceMethod']).get('CountDeclInstanceMethod', 0)
        private_methods = class_entity.metric(['CountDeclMethodPrivate']).get('CountDeclMethodPrivate', 0)
        return instance_methods - private_methods

    def test(self):
        print("Entity:", "Project")
        print(self.NOP)

    def get_class_entity(self, class_longname):
        for ent in self.db.ents(kindstring='class ~unknown'):
            if ent.longname() == class_longname:
                return ent
        return None

    def get_class_average(self, class_level_metric):
        scores = []
        for ent in self.db.ents(kindstring='class ~unknown'):
            if ent.kindname() == "Unknown Class":
                continue
            else:
                class_metric = class_level_metric(ent.longname())
                scores.append(class_metric)
        return sum(scores) / len(scores)

    def print_all(self):
        for k, v in sorted(self.metrics.items()):
            print(k, "=", v)


class QualityAttribute:
    def __init__(self, udb_path):
        """
        Implements Project Objectives due to QMOOD design metrics
        :param udb_path: The understand database path
        """
        self.udb_path = udb_path
        self.qmood = QMOODMetric(udb_path=udb_path)

    @property
    def reusability(self):
        """
        A design with low coupling and high cohesion is easily reused by other designs.
        :return: reusability score
        """
        score = -0.25 * self.qmood.DCC + 0.25 * self.qmood.CAMC + 0.5 * self.qmood.CIS + 0.5 * self.qmood.DSC
        return score

    @property
    def flexibility(self):
        """
        The degree of allowance of changes in the design.
        :return: flexibility score
        """
        score = 0.25 * self.qmood.DAM - 0.25 * self.qmood.DCC + 0.5 * self.qmood.MOA + 0.5 * self.qmood.NOP
        return score

    @property
    def understandability(self):
        """
        The degree of understanding and the easiness of learning the design implementation details.
        :return: understandability score
        """
        score = 0.33 * self.qmood.ANA + 0.33 * self.qmood.DAM - 0.33 * self.qmood.DCC + \
                0.33 * self.qmood.CAMC - 0.33 * self.qmood.NOP - 0.33 * self.qmood.NOM - \
                0.33 * self.qmood.DSC
        return score

    @property
    def functionality(self):
        """
        Classes with given functions that are publicly stated in interfaces to be used by others.
        :return: functionality score
        """
        score = 0.12 * self.qmood.CAMC + 0.22 * self.qmood.NOP + 0.22 * self.qmood.CIS + \
                0.22 * self.qmood.DSC + 0.22 * self.qmood.NOH
        return score

    @property
    def extendability(self):
        """
        Measurement of design's allowance to incorporate new functional requirements.
        :return: extendability
        """
        score = 0.5 * self.qmood.ANA - 0.5 * self.qmood.DCC + 0.5 * self.qmood.MFA + 0.5 * self.qmood.NOP
        return score

    @property
    def effectiveness(self):
        """
        Design efficiency in fulfilling the required functionality.
        :return: effectiveness score
        """
        score = 0.2 * self.qmood.ANA + 0.2 * self.qmood.DAM + 0.2 * self.qmood.MOA + 0.2 * \
                self.qmood.MFA + 0.2 * self.qmood.NOP
        return score


if __name__ == '__main__':
    db_path = 'D:/IdeaProjects/105_freemind/105_freemind.und'
    metric = QMOODMetric(db_path)
    # metric.print_all()
    # print(metric.MOA)
    # metric.test()
    qa = QualityAttribute(db_path)
    print('Reusability', qa.reusability)
    print('Flexibility', qa.flexibility)
    print('Understandability', qa.understandability)
    print('Functionality', qa.functionality)
    print('Extendability', qa.extendability)
    print('Effectiveness', qa.effectiveness)
