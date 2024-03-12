import tkinter as tk
from tkinter import ttk, messagebox
import requests

# Function to send a GET request to the server
def send_request(endpoint, params):
    try:
        response = requests.get(f"{url.get()}/{endpoint}", params=params)
        response_data = response.json()
        return response_data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send request: {e}")
        return None

# Specialized handling for the analytical_trends endpoint
def handle_analytical_trends():
    try:
        params = {
            "analysis": analysis_entry.get(),
            "screener": screener_entry.get(),
            "exchange": exchange_entry.get(),
            "interval": interval_entry.get()
        }
        response_data = send_request("analytical_trends", params)
        if response_data:
            min_price = float(min_price_entry.get())
            max_price = float(max_price_entry.get())
            filtered_stocks = [stock for stock in response_data.get(analysis_entry.get(), []) if min_price <= float(stock['price']) <= max_price]
    
            final_stocks = []
            for stock in filtered_stocks:
                analysis_params = {
                    "symbol": stock['symbol'],
                    "screener": screener_entry.get(),
                    "exchange": exchange_entry.get(),
                    "interval": "in_1_hour"
                }
                analysis_response = send_request("analysis", analysis_params)
                if analysis_response and analysis_response.get('summary', {}).get('RECOMMENDATION') == analysis_entry.get():
                    final_stocks.append(f"{stock['symbol']}: {stock.get('description', 'No description')}")
    
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, "\n".join(final_stocks))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send request: {e}")

# Function to handle the request based on user selection for other endpoints
def handle_request():
    if endpoint_combobox.get() == "analytical_trends":
        handle_analytical_trends()
    else:
        messagebox.showinfo("Info", "This demo specifically handles the analytical_trends endpoint. Please integrate other endpoints as needed.")

# Creating the main window
root = tk.Tk()
root.title("Analyse the Market")
root.geometry("700x500")

# Entry for the URL
tk.Label(root, text="Server URL:").grid(row=0, column=0, sticky="w")
url = tk.Entry(root)
url.grid(row=0, column=1, sticky="ew", columnspan=3)
url.insert(0, 'http://127.0.0.1:5000/')  # Default value

# Endpoint selection
tk.Label(root, text="Select Endpoint:").grid(row=1, column=0, sticky="w")
endpoint_options = ["analytical_trends"]  # Simplified for this demo
endpoint_combobox = ttk.Combobox(root, values=endpoint_options, state="readonly")
endpoint_combobox.grid(row=1, column=1, sticky="ew", columnspan=3)
endpoint_combobox.set("analytical_trends")  # Default selection

# Entries for parameters
tk.Label(root, text="Analysis Type:").grid(row=2, column=0, sticky="w")
endpoint_options = ["STRONG_BUY", "STRONG_SELL"]
analysis_entry = ttk.Combobox(root, values=endpoint_options, state="readonly")
analysis_entry.grid(row=2, column=1, sticky="ew")

tk.Label(root, text="Screener:").grid(row=3, column=0, sticky="w")
screener_entry = tk.Entry(root)
screener_entry.grid(row=3, column=1, sticky="ew")

tk.Label(root, text="Exchange:").grid(row=4, column=0, sticky="w")
exchange_entry = tk.Entry(root)
exchange_entry.grid(row=4, column=1, sticky="ew")

tk.Label(root, text="Interval:").grid(row=5, column=0, sticky="w")
endpoint_options = ["in_daily"]
interval_entry = ttk.Combobox(root, values=endpoint_options, state="readonly")
interval_entry.grid(row=5, column=1, sticky="ew")

# Entries for price filtering
tk.Label(root, text="Min Price:").grid(row=6, column=0, sticky="w")
min_price_entry = tk.Entry(root)
min_price_entry.grid(row=6, column=1, sticky="ew")

tk.Label(root, text="Max Price:").grid(row=7, column=0, sticky="w")
max_price_entry = tk.Entry(root)
max_price_entry.grid(row=7, column=1, sticky="ew")

# Button to send request
send_button = tk.Button(root, text="Send Request", command=handle_request)
send_button.grid(row=8, column=0, columnspan=4, sticky="ew")

# Text widget to display output
output_text = tk.Text(root, height=15)
output_text.grid(row=9, column=0, columnspan=4, sticky="nsew")

# Make the grid layout responsive
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(9, weight=1)

root.mainloop()
