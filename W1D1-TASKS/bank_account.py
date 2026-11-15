# Custom Exception

class InsufficientFundsError(Exception):
    pass


# Account Class

class Account:

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    # Deposit Method

    def deposit(self, amount):
        self.balance += amount
        print(self.owner, "deposited", amount, "-> balance", self.balance)

    # Withdraw Method

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance {self.balance}, requested {amount}"
            )

        self.balance -= amount

    # Show Method

    def show(self):
        print("Owner:", self.owner)
        print("Balance:", self.balance)


# Create Account Object

account = Account("Arun", 1000)

# Deposit Money

account.deposit(500)

# Withdraw Money

try:
    account.withdraw(2000)

except InsufficientFundsError as e:
    print("Error:", e)

finally:
    print("Transaction finished")
    