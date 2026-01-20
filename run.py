# =============================================================================
# run.py
# =============================================================================

"""
Main script to run the ASL translator
Choose what you want to do from the menu
"""

def show_menu():
    print("\n" + "="*60)
    print("ASL TRANSLATOR")
    print("="*60)
    print("\nWhat do you want to do?")
    print("1 - Collect training data")
    print("2 - Train the model")
    print("3 - Run live prediction")
    print("4 - Exit")
    print()

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            # data collection
            from collect_data import collect_all_letters
            collect_all_letters()
        
        elif choice == '2':
            # training
            from train_model import train
            train()
        
        elif choice == '3':
            # live demo
            import os
            if not os.path.exists(config.MODEL_FILE):
                print("\nModel not found!")
                print("You need to train the model first (option 2)")
            else:
                from live_demo import run_live_prediction
                run_live_prediction()
        
        elif choice == '4':
            print("\nBye!")
            break
        
        else:
            print("\nInvalid choice! Please enter 1, 2, 3, or 4")

if __name__ == "__main__":
    import config
    main()
