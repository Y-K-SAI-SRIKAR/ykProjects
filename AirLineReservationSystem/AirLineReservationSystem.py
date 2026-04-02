import random
import string

# File paths for storing data
FLIGHTS_FILE = "flights.txt"
BOOKINGS_FILE = "bookings.txt"

# Initialize files if they don't exist
def initialize_files():
    open(FLIGHTS_FILE, 'a').close()
    open(BOOKINGS_FILE, 'a').close()

# Load flights from a text file
def load_flights():
    flights = {}
    with open(FLIGHTS_FILE, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                flight_id, destination, seats, booked = line.split(',')
                flights[flight_id] = {
                    "destination": destination,
                    "seats": int(seats),
                    "booked": int(booked)
                }
    return flights

# Save flights to a text file
def save_flights(flights):
    with open(FLIGHTS_FILE, 'w') as file:
        for flight_id, details in flights.items():
            line = f"{flight_id},{details['destination']},{details['seats']},{details['booked']}\n"
            file.write(line)

# Load bookings from a text file
def load_bookings():
    bookings = {}
    with open(BOOKINGS_FILE, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                ref_id, user_name, flight_id = line.split(',')
                bookings[ref_id] = {"user": user_name, "flight_id": flight_id}
    return bookings

# Save bookings to a text file
def save_bookings(bookings):
    with open(BOOKINGS_FILE, 'w') as file:
        for ref_id, details in bookings.items():
            line = f"{ref_id},{details['user']},{details['flight_id']}\n"
            file.write(line)

# Admin functions
def add_flight():
    flights = load_flights()
    flight_id = input("Enter flight ID: ")
    destination = input("Enter destination: ")
    seats = int(input("Enter number of seats: "))
    flights[flight_id] = {"destination": destination, "seats": seats, "booked": 0}
    save_flights(flights)
    print("Flight added successfully.")

def remove_flight():
    flights = load_flights()
    flight_id = input("Enter flight ID to remove: ")
    if flight_id in flights:
        del flights[flight_id]
        save_flights(flights)
        print("Flight removed successfully.")
    else:
        print("Flight not found.")

# Function to view all available flights
def view_flights():
    flights = load_flights()
    if not flights:
        print("No flights available.")
    else:
        print("\n--- Available Flights ---")
        for flight_id, details in flights.items():
            available_seats = details['seats'] - details['booked']
            print(f"Flight ID: {flight_id}")
            print(f"Destination: {details['destination']}")
            print(f"Seats Available: {available_seats}/{details['seats']}")
            print("-----------------------")
        print()

# User functions
def book_flight():
    flights = load_flights()
    bookings = load_bookings()
    
    flight_id = input("Enter flight ID to book: ")
    if flight_id in flights:
        if flights[flight_id]["booked"] < flights[flight_id]["seats"]:
            user_name = input("Enter your name: ")
            ref_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            bookings[ref_id] = {"user": user_name, "flight_id": flight_id}
            flights[flight_id]["booked"] += 1
            save_flights(flights)
            save_bookings(bookings)
            print(f"Flight booked successfully. Reference ID: {ref_id}")
        else:
            print("No seats available.")
    else:
        print("Flight not found.")

def cancel_flight():
    bookings = load_bookings()
    flights = load_flights()
    
    ref_id = input("Enter your booking reference ID to cancel: ")
    if ref_id in bookings:
        flight_id = bookings[ref_id]["flight_id"]
        flights[flight_id]["booked"] -= 1
        del bookings[ref_id]
        save_flights(flights)
        save_bookings(bookings)
        print("Booking canceled successfully.")
    else:
        print("Booking not found.")

def print_ticket():
    bookings = load_bookings()
    flights = load_flights()
    
    ref_id = input("Enter your booking reference ID to print ticket: ")
    if ref_id in bookings:
        booking = bookings[ref_id]
        flight = flights[booking["flight_id"]]
        print("\n--- Ticket ---")
        print(f"Name: {booking['user']}")
        print(f"Flight ID: {booking['flight_id']}")
        print(f"Destination: {flight['destination']}")
        print(f"Reference ID: {ref_id}")
        print("--- End of Ticket ---\n")
    else:
        print("Booking not found.")

# Main function
def main():
    initialize_files()
    user_type = input("Enter 'admin' or 'user' to login: ").strip().lower()

    if user_type == 'admin':
        while True:
            print("\nAdmin Menu:")
            print("1. Add Flight")
            print("2. Remove Flight")
            print("3. View Flights")
            print("4. Logout")
            choice = input("Choose an option: ")
            if choice == '1':
                add_flight()
            elif choice == '2':
                remove_flight()
            elif choice == '3':
                view_flights()
            elif choice == '4':
                print("Logged out.")
                break
            else:
                print("Invalid option.")

    elif user_type == 'user':
        while True:
            print("\nUser Menu:")
            print("1. View Flights")
            print("2. Book Flight")
            print("3. Cancel Flight")
            print("4. Print Ticket")
            print("5. Logout")
            choice = input("Choose an option: ")
            if choice == '1':
                view_flights()
            elif choice == '2':
                book_flight()
            elif choice == '3':
                cancel_flight()
            elif choice == '4':
                print_ticket()
            elif choice == '5':
                print("Logged out.")
                break
            else:
                print("Invalid option.")

    else:
        print("Invalid user type.")

if __name__ == "__main__":
    main()
