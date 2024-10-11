class Existentials:
    _instance = None
    no_of_existentials = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Existentials, cls).__new__(cls)

        return cls._instance

    def getNoOfExistentials(self):
        return self.no_of_existentials

    def increaseNoOfExistentials(self):
        self.no_of_existentials += 1

    def increaseAndGetExistential(self):
        self.increaseNoOfExistentials()
        return self.no_of_existentials