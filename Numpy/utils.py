import numpy as np
def calculate_revenue(data):
    price = data[:, 1]
    units_sold = data[:, 4]

    revenue = price * units_sold
    return revenue


def normalize_data(values):
    
    return (values - np.min(values)) / (np.max(values) - np.min(values))


def rank_top_sales(revenue, top_n=5):
    
    return np.argsort(revenue)[-top_n:][::-1]


def predict_next_year_sales(units_sold, growth_rate=0.05):
    
    return units_sold * (1 + growth_rate)


def statistics(revenue):
    
    print("\n Revenue Statistics")
    print("Total Revenue:", np.sum(revenue))
    print("Average Revenue:", np.mean(revenue))
    print("Maximum Revenue:", np.max(revenue))
    print("Minimum Revenue:", np.min(revenue))
    print("Standard Deviation:", np.std(revenue))