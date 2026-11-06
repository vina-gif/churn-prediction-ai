import requests

URL = "http://127.0.0.1:5000/predict"

# Customer 1: new customer, high churn risk
customer_1 = {
    "tenure": 3,
    "MonthlyCharges": 95.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "SeniorCitizen": 0,
    "PhoneService": "Yes",
    "OnlineSecurity": "No",
    "TechSupport": "No",
}

# Customer 2: loyal customer, low churn risk
customer_2 = {
    "tenure": 60,
    "MonthlyCharges": 40.0,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Bank transfer (automatic)",
    "PaperlessBilling": "No",
    "SeniorCitizen": 0,
    "PhoneService": "Yes",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
}

# Send both customers to the API and print the results
print("Customer 1 (new, month-to-month):")
print(requests.post(URL, json=customer_1).json())

print("\nCustomer 2 (loyal, two-year contract):")
print(requests.post(URL, json=customer_2).json())