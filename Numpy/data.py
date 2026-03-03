import numpy as np

def generate_data(records=1000):
   
    np.random.seed(42)

    car_ids = np.arange(1, records + 1)
    price = np.random.randint(500000, 2500000, records)
    engine_size = np.random.randint(1000, 3000, records)
    car_age = np.random.randint(0, 11, records)
    units_sold = np.random.randint(1, 51, records)

    data = np.column_stack((car_ids, price, engine_size, car_age, units_sold))

    return data