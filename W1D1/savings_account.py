# Custom Exception

class InsufficientFundsError(Exception):
    pass


# Parent Class

class Account:

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount

    def show(self):
        print("Owner:", self.owner)
        print("Balance:", self.balance)

    # Bonus: String Representation

    def __str__(self):
        return f"Owner: {self.owner}, Balance: {self.balance}"


# Child Class

class SavingsAccount(Account):

    def __init__(self, owner, balance=0, min_balance=500):
        super().__init__(owner, balance)
        self.min_balance = min_balance

    # Method Overriding

    def withdraw(self, amount):

        if self.balance - amount < self.min_balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount

        print(self.owner, "withdrew", amount, "-> balance", self.balance)


# Create Priya Account

priya = SavingsAccount("Priya", 1500)


# First Withdrawal

try:
    priya.withdraw(1200)

except InsufficientFundsError as e:
    print("Error:", e)


# Second Withdrawal

priya.withdraw(800)


# Check Inheritance

print(isinstance(priya, Account))


# Bonus

print(priya)


