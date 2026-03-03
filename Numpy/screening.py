import numpy as np
from data import generate_data
from utils import (
    calculate_revenue,
    normalize_data,
    rank_top_sales,
    predict_next_year_sales,
    statistics
)


def main():

    print(" Car Sales Prediction Using NumPy\n")

 
    data = generate_data(records=1000)
    print("Dataset Shape:", data.shape)

    revenue = calculate_revenue(data)

    normalized_revenue = normalize_data(revenue)
    print("\nMemory Usage (MB):", normalized_revenue.nbytes / 1024 / 1024)

    top_sales = rank_top_sales(revenue)
    print("\n Top Sales Indexes:", top_sales)
    print(" Top Revenues:", revenue[top_sales])

    units_sold = data[:, 4]
    predicted_units = predict_next_year_sales(units_sold)
    predicted_revenue = data[:, 1] * predicted_units

    print("\n Predicted Total Revenue Next Year:", np.sum(predicted_revenue))

  
    statistics(revenue)

    print("\n Car Sales Analysis Completed Successfully")


if __name__ == "__main__":
    main()