import os
import sys
import re
import pickle


class Person:
    def __init__(self, last, first, mi, id, phone):
        self.last = last
        self.first = first
        self.mi = mi
        self.id = id
        self.phone = phone

    def display(self):
        print("Employee ID: " + self.id)
        print("\t{} {} {}".format(self.last, self.mi, self.first))
        print("\t{}".format(self.phone))


def process_text(unprocessed_text):
    employees = {}

    data = unprocessed_text.split('\n')

    for i in range(len(data)):
        tokens = data[i].split(',')

        last = tokens[0]
        first = tokens[1]
        mi = tokens[2]
        id = tokens[3]
        phone = tokens[4]

        # modify last name to be capital case
        last = last.capitalize()

        # modify first name to be capital case
        first = first.capitalize()

        # modify middle initial to be a single upper case letter, if necessary. Use ‘X’ as a middle
        # initial if one is missing.
        mi = 'X' if mi == '' else mi[0].upper()

        # modify id if necessary, using regex. The id should be 2 letters followed by 4 digits. If an
        # id is not in the correct format, output an error message, and allow the user to re-enter a
        # valid ID
        id_pattern = re.compile(r'[A-Za-z]{2}\d{4}')

        while not re.match(id_pattern, id):
            print("Invalid ID: " + id)
            print("ID is two letters followed by 4 digits")
            id = input("Please enter a valid id: ")

        # modify phone number, if necessary, to be in form 999-999-9999. Use regex.
        phone = '%s-%s-%s' % tuple(re.findall(r'\d{4}$|\d{3}', phone))

        # create a new person
        employee = Person(first, last, mi, id, phone)

        # place employee in a dict where the key is the employee's ID.
        # ensure that there are no repeat employees who have the same ID
        if id in employees:
            print("Error: Employee ID is repeated multiple times in input file")
        else:
            employees[id] = employee

    return employees


if __name__ == '__main__':
    # if the user didn't specify a relative path, print an error
    if len(sys.argv) < 2:
        print("Error: Relative path not specified.")
        exit()
    else:
        fp = sys.argv[1]

        # read from file specified by user's relative path
        try:
            with open(os.path.join(os.getcwd(), fp), 'r') as f:
                next(f) # skips first line
                text_in = f.read()
            employees = process_text(text_in)

            # pickle the dictionary with the employees
            pickle.dump(employees, open('dict.pkl', 'wb'))

            # unpickle the dictionary with the employees
            unpickled = pickle.load(open('dict.pkl', 'rb'))

            # print the employee list
            print("Employee List: ")
            for employees in unpickled.values():
                employees.display()

        except FileNotFoundError:
            print("Error: File path not specified correctly")
            exit()