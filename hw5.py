import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_data(filename):
    # Load the dataset
    df = pd.read_csv(filename)

    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.plot(df["year"], df["days"], marker="o", linestyle="-")
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.title("Lake Mendota Ice Days Over Time")

    # Set x-axis ticks to display as integers - used deepseek for this as was confused how to stop
    # floating points. prompt "how to get rid of floating point values in plot"
    plt.xticks(df["year"].astype(int))  # Ensure years are displayed as integers

    # Save the figure
    plt.savefig("data_plot.jpg")


def normalize_data(filename):
    df = pd.read_csv(filename)

    # Extract the year column
    x = df["year"].values

    # Min and max
    m = np.min(x)
    M = np.max(x)

    # Min-max normalization
    x_normalized = (x - m) / (M - m)

    # Augment the normalized data with ones for bias term
    X_normalized = np.column_stack((x_normalized, np.ones_like(x_normalized)))

    Y = df["days"].values

    print("Q2:")
    print(X_normalized)

    return X_normalized, Y, m, M


def closed_form_solution(X_normalized, Y):
    # Compute the closed-form solution (w, b) | used gpt for this as i was unfamialr.
    #prompt: "help me w this [mehtod name]. gave code bleow
    weights = np.linalg.inv(X_normalized.T @ X_normalized) @ X_normalized.T @ Y

    w, b = weights[0], weights[1]

    # Print the result
    print("Q3:")
    print(weights)

    return weights


def gradient_descent(X_normalized, Y, learning_rate, iterations):
    n = len(Y)
    weights = np.zeros(2)  # Initialize w and b to 0
    loss_history = []

    print("Q4a:")
    for t in range(iterations):
        # Print weights before the first update
        # gpt fixed an error here. was printing incorrectly so helped w formatting.
        if t % 10 == 0:
            print(weights)

        y_pred = X_normalized @ weights  # Compute predictions
        gradient = (1 / n) * X_normalized.T @ (y_pred - Y) #Compute gradient used gpt again, same as clsoed form method
        weights -= learning_rate * gradient  # Update weights

        # Compute MSE loss
        loss = (1 / (2 * n)) * np.sum((y_pred - Y) ** 2)
        loss_history.append(loss)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(iterations), loss_history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.legend()
    plt.savefig("loss_plot.jpg")  # Save loss plot

    return weights


def predict_ice_days(weights, m, M):
    year = 2023  # Year to predict for

    x_normalized = (year - m) / (M - m)

    y_hat = weights[0] * x_normalized + weights[1]  # Prediction
    y_hat_rounded = round(y_hat, 13)
    print("Q5: " + str(y_hat_rounded)) #i am rounding to match output



def interpret_weight(w):
    # Checking sign of w
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="

    print("Q6a: " + symbol)


def interpret_model(w):
    print("Q6b: If w > 0, frozen days increase over time. If w < 0, frozen days decrease, possibly due to climate change. If w = 0, frozen days remain constant.")


def predict_no_freeze_year(w, b, m, M):
    # Div by 0 undefined - check:
    if w == 0:
        print("Q7a: Undefined (w = 0, cannot predict)")
        return

    # Compute x_star using the formula
    x_star = m + (-b / w) * (M - m)

    print("Q7a: " + str(x_star))
    print("Q7b: The prediction assumes a linear trend, but climate change effects may not be linear due to external factors.")

def main():
    # Read CLA, will need q4 onwards
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])
    plot_data(filename)

    X_normalized, Y, m, M = normalize_data(filename)
    closed_form_weights = closed_form_solution(X_normalized, Y)

    w, b = closed_form_weights[0], closed_form_weights[1]  # For interpreting w (q6)

    gd_weights = gradient_descent(X_normalized, Y, learning_rate, iterations) #ignore usage, need to do for q4 onwards

    print("Q4b:", learning_rate)
    print("Q4c:", iterations)
    print("Q4d: Started with a small learning rate to ensure stability and avoid overshooting the optimal weights.")

    predict_ice_days(closed_form_weights, m, M)

    interpret_weight(w)
    interpret_model(w)

    predict_no_freeze_year(w, b, m, M)

#for my running
if __name__ == "__main__":
    main()